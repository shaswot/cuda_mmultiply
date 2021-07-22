#include <istream>
#include <iostream>
#include <fstream>

#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

// https://github.com/arpaka/mnist-loader
// #include "../include/mnist_loader.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>

template <class _Tp>
xt::xarray<_Tp> matmul(xt::xarray<_Tp> a, 
                       xt::xarray<_Tp> b) noexcept(false)
{
//     if (a.shape().size() != b.shape().size()) {
//         throw std::runtime_error("Shape mismatching!");
//     }
    
    const unsigned int n = a.shape()[0]; // a rows
    const unsigned int m = a.shape()[1]; // a cols
    const unsigned int p = b.shape()[1]; // b cols

//     std::cout<<"n = "<<n<<std::endl;
//     std::cout<<"m = "<<m<<std::endl;
//     std::cout<<"p = "<<p<<std::endl;
    
    xt::xarray<double>::shape_type shape = {n,p};
    xt::xarray<float> out = xt::zeros<_Tp>(shape);

//     std::cout<<out<<std::endl;

    for (auto i = 0; i < n; ++i){
        for (auto j = 0; j < p; ++j){
            for (auto k = 0; k < m; ++k){
                out(i,j) += a(i,k) * b(k,j);
            }
        }
    }
    return out;
}


int main()
{
    auto w_mat = xt::load_npy<float>("../data/random_array.npy");
    auto i_vec = xt::load_npy<float>("../data/random_input.npy");
    auto i_mat = xt::load_npy<float>("../data/random_input_mat.npy");

    i_vec.reshape({-1, 1});
    
    auto y1 = matmul<float>(w_mat, i_vec);
    auto y2 = matmul<float>(w_mat, i_mat);
    std::cout << "Matrix" << std::endl << w_mat << std::endl;
    std::cout << "Vector" << std::endl << i_vec << std::endl;
    std::cout << "Matrix * Vector" << std::endl<< y1 <<std::endl;
    std::cout<<"******************************" <<std::endl;
    
    std::cout << "Matrix A" << std::endl << w_mat << std::endl;
    std::cout << "Matrix B" << std::endl << i_mat << std::endl;
    std::cout << "Matrix * Matrix" << std::endl << y2 <<std::endl;

    
    return 0;
}