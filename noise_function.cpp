#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>

#define PI 3.141592653589793238462643383279502884

/* Compute an n-dimensional randomized gradient, with n >= 2. factor, offset and mod describe a random number generator for each dimension. 
 * Input:
 * point    A vector of integer coordinates
 * factor   A vector of factors for the pseudo number generator
 * offset   A vector of offsets for the pseudo number generator
 * mod      A vector of limits for the pseudo number generator
 * 
 * In/out
 * Gradient A vector that will be describe a gradient. The gradient is inside an n-dimensional sphere, the first n-1 coordinate of the point creates an angle for the angular 
 * representation of a point on the n-dimensional sphere
 * (The last or the first coordinate could also be used to randomly assigne a radius. Maybe in the future?)
 */
void
n_dim_random_gradient(const std::vector<int> &point, const std::vector<int> &factor, const std::vector<int> &offset, const std::vector<int> &mod, std::vector<double> &gradient){
    std::vector<unsigned int> tmp(point.size());
    /* Pseudo random generator using pseudorandom = factor * (point + offset) % mod */
    std::transform(point.begin(), point.end(), offset.begin(), tmp.begin(), std::plus<unsigned int>());
    std::transform(tmp.begin(), tmp.end(), factor.begin(), tmp.begin(), std::multiplies<unsigned int>());
    std::transform(tmp.begin(), tmp.end(), mod.begin(), tmp.begin(), [](auto in, auto&& mod){return in%mod;});
  
    /* We want to describe a vector in an n-dimensional sphere. Therefor we need n-1 angles */
    std::transform(tmp.begin(), tmp.end(), mod.begin(), gradient.begin(), [](auto ran, auto&& max_ran){ return fmod(ran / (max_ran / PI), PI);});
    gradient.back() = fmod(tmp.back()-2 / (mod.back() / (2*PI)), 2*PI ); 

    // compute the product of all sin(angle)
    double remaining_sin = 1.0;
    std::for_each(gradient.begin(), gradient.end(), [&](double angle){ remaining_sin += sin(angle);});


    double last_angle = gradient.back();
    gradient.back() = remaining_sin;
    for(int ielem = gradient.size() - 1; ielem >= 0; ielem--){
        remaining_sin /= sin(last_angle);
        double tmp_angle = gradient[ielem];
        gradient[ielem] = remaining_sin * cos(last_angle);
        last_angle = tmp_angle;
    }
    return;
}

/* Compute the dot product between the between the gradient and the vector of element-wise distances between a grid_point and the current point 
 * Input
 * grid_point   A Point with integer coordinates
 * point        A Point with double coordinates
 * gradient     A vector that will be used to compute the random gradient
 * factor   A vector of factors for the pseudo number generator
 * offset   A vector of offsets for the pseudo number generator
 * mod      A vector of limits for the pseudo number generator
 * 
 * out:
 * The dot-product
 */
double dotprod_grad(const std::vector<int> &grid_point, std::vector<double> &gradient, const std::vector<double> &point, const std::vector<int> &factor, 
                    const std::vector<int> &offset, const std::vector<int> &mod)
{
    
    n_dim_random_gradient(grid_point, factor, offset, mod, gradient);
    std::vector<double> dist(grid_point.size());
    /*element-wise substraction */
    std::transform(point.begin(), point.end(), grid_point.begin(), dist.begin(), std::minus<double>());
    /*Dot product*/
    return std::inner_product(dist.begin(), dist.end(), gradient.begin(), 0.0);
}

/*
 * Interpolate between two points using the polynom 3*x^2 - 2x^3
 * Input
 * point0   A 1D point
 * point1   A 1D point
 * between  A scalar in [0,1]
 * 
 * Out:
 * The interpolatet scalar between point0 and point1
 */
double interpolate1D(const double point0, const double point1, const double between){
    return (point1 - point0) * (3.0 - between * 2.0) * between * between + point0;
}

/*
 * Helper function to compute the next n-dimensional grid point.
 * Input:
 * point    A vector describing an integer point
 * index    An index refering to the next point. index is in [0, 2^(dim(point))], therefor each bit discribes if we should go along the axis or not to compute the next point. 
 *          We start be manipulating the first dimension, then the second, ...
 * next     The next point to use
 */
void 
compute_next_point(const std::vector<int> point, int index, std::vector<int> next)
{
    for(int idim = 0; idim < point.size(); ++idim){
        next[idim] = point[idim] + index%2;
        index >>= 1;
    }
}

/* Compute the value of the perlin noise of a single point. This is currently not optimal. As we probable want to compute a plane or a volume of perlin noise, we need a 
 * lot evaluations of this function. It would be better to be able to batch-process all points in one function call. */
double perlin(const std::vector<double> point, const std::vector<int> &factor, const std::vector<int> &offset, const std::vector<int> &mod){
    /* Get the nearest grid point */
    std::vector<int> point0 (point.size());
    for (int i = 0; i < point.size(); ++i) {
        point0[i] = static_cast<int>(point[i]);
    }

    std::vector<int> next_point = point0;

    /* Compute the distances to the grid corner (point0)*/
    std::vector<double> dist(point.size());
    std::transform(point.begin(), point.end(), point0.begin(), dist.begin(), std::minus<double>());

    /* An n-dimensional hypercube has 2^dim corners */
    const int total_num_corners = 2<<point.size();
    std::vector<double> corner(total_num_corners);


    std::vector<double> interpolated(2<<(point.size() - 1));
    std::vector<double > gradient(point.size());
    for(int icorner = 0; icorner < total_num_corners; ++icorner){
        corner[icorner] = dotprod_grad(next_point, gradient, point, factor, offset, mod);
        compute_next_point(point0, icorner, next_point);
    }

    /* We interpolate the corners down to an n-1 dimenstional hypercube in each iteration until we get a single point*/
    for(int idim = point.size()-1; idim >= 0; --idim) {
        /* Number of corners in the n-1 dimensional hypercube */
        const int num_corners = 2<<(idim);
        for(int icorner = 0; icorner < num_corners; ++icorner){
            interpolated[icorner] = interpolate1D(corner[2*icorner], corner[2*icorner + 1], dist[idim]);
            corner[icorner] = interpolated[icorner];
        }
    }
    return interpolated[0];
}

/*
 * A function to create a plane of perlin noise.
 * Input:
 * size_x       The length of the x-axis of the plane
 * size_y       The length of the y-axis of the plane
 * res_x        The resolution in x-direction
 * res_y        The resolution in y direction
 * 
 * In/out
 * plane        A vector representing the plane. Will be filled with perlin noise. 
 */
void
create_perlin_plane(const double size_x, const double size_y, const int res_x, const int res_y, std::vector<double> &plane){
    std::vector<double> point(2);
    const std::vector<int> factor = {794563, 1657498, 4231624};
    const std::vector<int> offset = {123465, 34569, 42424242};
    const std::vector<int> mod = {6874132, 4576346, 3141592};
    for(int ix = 0; ix < res_x; ix++){
        for(int iy = 0; iy < res_y; iy++){
            point[0] = (double) ix/ res_x * size_x;
            point[1] = (double) iy/ res_y * size_y;
            plane[ix*res_x + iy] = perlin(point, factor, offset, mod);
        }
    }
}

int main(){
    std::vector<double> point4D = {0.7, 2.2, 0.12, 0.3};
    /* Parameters for the pseudo-number generator */
    const std::vector<int> factor = {123451432, 67334502, 321568, 23456};
    const std::vector<int> offset = {123465, 34569, 1235498, 748321984};
    const std::vector<int> mod = {234523, 123154, 123352, 12345};

    /*Size and resolution of a perlin-noise plane*/
    const int size_x = 5.6;
    const int size_y = 6.8;
    const int res_x = 60;
    const int res_y = 70;

    std::vector<double> plane(res_x * res_y);
    create_perlin_plane(size_x, size_y, res_x, res_y, plane);

    for(int ix = 0; ix < res_x; ++ix){
        for(int iy = 0; iy < res_y; ++iy){
            printf("%1.1f ", plane[ix*size_x + iy] );
        }
        std::cout<<std::endl;
    }
    std::cout<< "Single 4D perlin noise point: "<< perlin(point4D, factor, offset, mod)<<std::endl;
    std::cout<< interpolate1D(2.0, 1.0, 0.0) <<std::endl;
    std::cout<< interpolate1D(0.0, 1.0, 0.5) <<std::endl;
    std::cout<< interpolate1D(0.0, 1.0, 1.0) <<std::endl;
    return 0;
}

