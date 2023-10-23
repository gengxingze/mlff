/* -*- c++ -*- ----------------------------------------------------------
     PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlff, PairMlff);
// clang-format on
#else



#ifndef LMP_PAIR_MLFF_H
#define LMP_PAIR_MLFF_H

#include "pair.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>


namespace LAMMPS_NS {

    class PairMlff : public Pair {
        public:
            PairMlff(class LAMMPS *);
            ~PairMlff() override;

            void compute(int, int) override;
            void settings(int, char **) override;
            void coeff(int, char **) override;
            double init_one(int, int) override;
            void init_style() override;

        protected:
            virtual void allocate();
        
        private:
            torch::jit::script::Module module;
            torch::Device device = torch::kCPU;
            torch::Dtype dtype = torch::kFloat32;
            std::vector<int> type_map;
            double cutoff;
            int max_neighbor;
            std::string model_name;

    };

}
#endif
#endif
