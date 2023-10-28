#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>

#include "pair_mlff.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMlff::PairMlff(LAMMPS *lmp) : Pair(lmp)
{
	writedata = 1;
}

PairMlff::~PairMlff()
{
    if (allocated) 
    {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMlff::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes ;
    memory->create(setflag, np1 + 1, np1 + 1, "pair:setflag");
    for (int i = 1; i <= np1; i++)
        for (int j = i; j <= np1; j++) setflag[i][j] = 0;
    memory->create(cutsq, np1 + 1, np1 + 1, "pair:cutsq");

}

static bool is_key(const std::string& input) {
    std::vector<std::string> keys;
    keys.push_back("out_freq");
    keys.push_back("out_file");

    for (int ii = 0; ii < keys.size(); ++ii) {
        if (input == keys[ii]) {
            return true;
        }
    }
    return false;
}


/* ----------------------------------------------------------------------
   global settings pair_style 
------------------------------------------------------------------------- */

void PairMlff::settings(int narg, char** arg)
{
    if (narg <= 0) error->all(FLERR, "Illegal pair_style command");
    std::vector<std::string> models;

    int iarg;
    while (iarg < narg) {
        if (is_key(arg[iarg])) {
            break;
        }
        iarg++;
    }

    for (int ii = 0; ii < iarg; ++ii) {
        models.push_back(arg[ii]);
    }

    if (models.size() == 1) {
        try
        {
            std::string model_file = models[0];
            torch::jit::getExecutorMode() = false;
            module = torch::jit::load(model_file);
            if (torch::cuda::is_available()) { device = torch::kCUDA; }
            if (true) { dtype = torch::kFloat32; }
            module.to(device, dtype);
            module.eval();
            cutoff = module.attr("Rmax").toDouble();
            max_neighbor = module.attr("max_neighbor").toInt();
            torch::Device device = torch::kCPU;
            utils::logmesg(lmp, "Load model successful !----> %s", model_file);
            utils::logmesg(lmp, "INFO IN MLFF-MODEL---->>");
            utils::logmesg(lmp, "\tModel type:   %f",cutoff);
            utils::logmesg(lmp, "\tcutoff :      %f",cutoff);
            utils::logmesg(lmp, "\tmax_neighbor: %d", max_neighbor);

        }
        catch (const c10::Error e)
        {
            std::cerr << "Failed to load model :" << e.msg() << std::endl;
        }
    }

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs pair_coeff 
------------------------------------------------------------------------- */

void PairMlff::coeff(int narg, char** arg)
{
    int ntype = atom->ntypes;
    if (!allocated) { allocate(); }

    // pair_coeff * * 
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    int count = 0;
    for(int i = ilo; i <= ihi; i++) {
        for(int j = MAX(jlo,i); j <= jhi; j++) 
        {
            setflag[i][j] = 1;
            count++;
        }
    }

    auto type_map_module = module.attr("type_map").toList();
    if (ntype > narg - 2 )
    {
        error->all(FLERR, "Element mapping not fully set");
    }
    for (int ii = 2; ii < narg; ++ii) {
        int temp = std::stoi(arg[ii]);
        auto iter = std::find(type_map_module.begin(), type_map_module.end(), temp);   
        if (iter != type_map_module.end() || arg[ii] == 0)
        {
            type_map.push_back(temp);
        }
        else
        {
            error->all(FLERR, "This element is not included in the machine learning force field");
        }
    }

   if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMlff::init_one(int i, int j)
{
    //if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set"); 

    return cutoff;
}


void PairMlff::init_style()
{
    // Using a nearest neighbor table of type full
    neighbor->add_request(this, NeighConst::REQ_FULL);
}
/* ---------------------------------------------------------------------- */

void PairMlff::compute(int eflag, int vflag)
{

    auto t1 = std::chrono::high_resolution_clock::now();
    if (eflag || vflag) ev_setup(eflag, vflag);
    bool do_ghost = true;
    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    int newton_pair = force->newton_pair;
    int nlocal = atom->nlocal;
    int nghost = 0;
    if (do_ghost) { nghost = atom->nghost; }
    int n_all = nlocal + nghost;

    int *ilist, *jlist, *numneigh, **firstneigh;
    int inum, jnum, itype, jtype;
    double dx, dy, dz, rsq, rij;
    
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    auto t2 = std::chrono::high_resolution_clock::now();

    double* ptrImagedR = (double*)malloc(sizeof(double) * (inum * max_neighbor * 4));
    memset(ptrImagedR, 0, sizeof(double) * (inum * max_neighbor * 4));
    int* ptrImagetype = (int*)malloc(sizeof(int) * inum);
    memset(ptrImagetype, 0, sizeof(int) * inum);
    int* ptrneighbor_list = (int*)malloc(sizeof(int) * (inum * max_neighbor));
    memset(ptrneighbor_list, 0, sizeof(int) * (inum * max_neighbor));
    int* ptrneighbor_type = (int*)malloc(sizeof(int) * (inum * max_neighbor));
    memset(ptrneighbor_type, 0, sizeof(int) * (inum * max_neighbor));
    // double ptrImagedR[1][inum][max_neighbor][4]; //error in cpp, stackoverflow in c;
    auto t3 = std::chrono::high_resolution_clock::now();

    std::vector<int> use_type(n_all);
    
    for (int ii = 0; ii < n_all; ii++)
    {
        use_type[ii] = type_map[type[ii] - 1];
    }

    double rc2 = cutoff * cutoff;
    auto t4 = std::chrono::high_resolution_clock::now();

    for (int ii = 0; ii < inum; ii++)
    {
        int i = ilist[ii];
        itype = use_type[i];
        jlist = firstneigh[i];
        ptrImagetype[ii] = itype;
        int kk = 0;
        for (int jj = 0; jj < numneigh[i]; jj++)
        {
            int j = jlist[jj];
            dx = x[i][0] - x[j][0];
            dy = x[i][1] - x[j][1];
            dz = x[i][2] - x[j][2];
            rsq = dx * dx + dy * dy + dz * dz;
            if (rsq <= rc2){
                if (kk < max_neighbor){
                    rij = sqrt(rsq);
                    int id_temp = ii * max_neighbor * 4 + kk * 4;
            	    ptrImagedR[id_temp] = rij;
                    ptrImagedR[id_temp + 1] = dx;
                    ptrImagedR[id_temp + 2] = dy;
                    ptrImagedR[id_temp + 3] = dz;
                    ptrneighbor_list[ii * max_neighbor + kk] = j + 1;
                    ptrneighbor_type[ii * max_neighbor + kk] = use_type[j];
                    kk = kk + 1;
                }
                else{
                    error->all(FLERR, "The maximal nearest neighbor you set may not be enough ");
                }
	        }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    torch::Tensor Imagetype = torch::from_blob(&ptrImagetype[0], { 1,inum }, torch::TensorOptions().dtype(torch::kInt)).to(device);
    torch::Tensor neighbor_list = torch::from_blob(&ptrneighbor_list[0], { 1,inum, max_neighbor}, torch::TensorOptions().dtype(torch::kInt)).to(device);
    torch::Tensor neighbor_type = torch::from_blob(&ptrneighbor_type[0], { 1,inum, max_neighbor }, torch::TensorOptions().dtype(torch::kInt)).to(device);
    torch::Tensor ImagedR = torch::from_blob(&ptrImagedR[0], { 1, inum, max_neighbor, 4}, torch::TensorOptions().dtype(torch::kFloat64)).to(device,dtype);
    //std::cout << neighbor_list[0][0] << std::endl;
    auto t6 = std::chrono::high_resolution_clock::now();
    auto output = module.forward({ Imagetype, neighbor_list, neighbor_type, ImagedR , nghost}).toTuple();
    auto t7 = std::chrono::high_resolution_clock::now();
    torch::Tensor Etot = output->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor Ei = output->elements()[1].toTensor().to(torch::kCPU);
    torch::Tensor Force = output->elements()[2].toTensor().to(torch::kCPU);



    // get force
    auto F_ptr = Force.accessor<float, 3>();
    auto Ei_ptr = Ei.accessor<float, 3>();


    for (int i = 0; i < inum + nghost; i++)
    {
        f[i][0] += F_ptr[0][i][0];
        f[i][1] += F_ptr[0][i][1];
        f[i][2] += F_ptr[0][i][2];
    }

    // get energy
    if (eflag)  eng_vdwl = Etot[0][0].item<double>();

    if (eflag_atom)
    {
        for (int ii = 0; ii < inum; ii++) {
            eatom[ii] = Ei_ptr[0][ii][0];
        }
    }
    // If virial needed calculate via F dot r.
    if (vflag_fdotr) virial_fdotr_compute();

    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "t1 " << (t2 - t1).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t2 " << (t3 - t2).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t3 " << (t4 - t3).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t4 " << (t5 - t4).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t5 " << (t6 - t5).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t6 " << (t7 - t6).count() * 0.000001 << "ms" << std::endl;
    std::cout << "t7 " << (t8 - t7).count() * 0.000001 << "ms" << std::endl;

}
