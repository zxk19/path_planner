#ifndef SMOOTHER_COST_FUNCTION_H
#define SMOOTHER_COST_FUNCTION_H

#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <queue>
#include <utility>
#include <nav_msgs/OccupancyGrid.h>

#include "node3d.h"
#include "ceres/ceres.h"
#include "Eigen/Core"
#include "constants.h"
#include "dynamicvoronoi.h"

#define EPSILON 0.0001

namespace HybridAStar
{

/**
 * @struct HybridAStar::UnconstrainedSmootherCostFunction
 * @brief Cost function for path smoothing with multiple terms
 * including curvature, smoothness, collision, voronoi, and avoid obstacles.
 */
class UnconstrainedSmootherCostFunction : public ceres::FirstOrderFunction
{
public:
  /**
   * @brief A constructor for UnconstrainedSmootherCostFunction
   * @param original_path Original unsmoothed path to smooth
   * @param voronoi A voronoi field of the map to get values for nearest obstacle and voronoi edge
   * @param params The smoother parameters
   */
  UnconstrainedSmootherCostFunction(
    std::vector<Eigen::Vector2d> * original_path,
    std::vector<bool> * prims,
    DynamicVoronoi & voronoi,
    const Constants::SmootherParams & params)
  : _original_path(original_path),
    _prims(prims),
    _num_params(2 * original_path->size()),
    _voronoi(&voronoi),
    _params(params)
  {
  }

  /**
   * @struct CurvatureComputations
   * @brief Cache common computations between the curvature terms to minimize recomputations
   */
  struct CurvatureComputations
  {
    /**
     * @brief A constructor for nav2_smac_planner::CurvatureComputations
     */
    CurvatureComputations()
    {
      valid = true;
    }

    bool valid;
    /**
     * @brief Check if result is valid for penalty
     * @return is valid (non-nan, non-inf, and turning angle > max)
     */
    bool isValid()
    {
      return valid;
    }

    Eigen::Vector2d delta_xi{0.0, 0.0};
    Eigen::Vector2d delta_xi_p{0.0, 0.0};
    double delta_xi_norm{0};
    double delta_xi_p_norm{0};
    double delta_phi_i{0};
    double turning_rad{0};
    double ki_minus_kmax{0};
  };

  /**
   * @brief Smoother cost function evaluation
   * @param parameters X,Y pairs of points
   * @param cost total cost of path
   * @param gradient of path at each X,Y pair from cost function derived analytically
   * @return if successful in computing values
   */
  virtual bool Evaluate(
    const double * parameters,
    double * cost,
    double * gradient) const
  {
    Eigen::Vector2d xi;
    Eigen::Vector2d xi_p1;
    Eigen::Vector2d xi_m1;
    Eigen::Vector2d xi_p2;
    Eigen::Vector2d xi_m2;
    uint x_index, y_index;
    cost[0] = 0.0;
    double cost_raw = 0.0;
    double grad_x_raw = 0.0;
    double grad_y_raw = 0.0;
    double obs_dist = 0.0;
    double voronoi_dist = 0.0;
    Eigen::Vector2d closest_voronoi_pt;
    Eigen::Vector2d closest_obstacle;
    float totalWeight = _params.smooth_weight + _params.distance_weight + _params.obstacle_weight + _params.voronoi_weight + _params.curvature_weight;


    // cache some computations between the residual and jacobian
    CurvatureComputations curvature_params;

    for (int i = 0; i != NumParameters() / 2; i++) {
      x_index = 2 * i;
      y_index = 2 * i + 1;
      gradient[x_index] = 0.0;
      gradient[y_index] = 0.0;

      xi = Eigen::Vector2d(parameters[x_index], parameters[y_index]);
      xi_p1 = Eigen::Vector2d(parameters[x_index + 2], parameters[y_index + 2]);
      xi_m1 = Eigen::Vector2d(parameters[x_index - 2], parameters[y_index - 2]); 

      // Make sure the start and end points are not changed.
      if (i < 1 || i >= NumParameters() / 2 - 1) {
        continue;
      }
      // Make sure the cusp points are not changed
      if (_prims->at(i)) {
        //std::cout << "******************CUSP AT: " << i << "Coordinates are: " << xi << std::endl;
        continue;}

      /*********** compute cost ***********/
      addSmoothingResidual(_params.smooth_weight, xi, xi_p1, xi_m1, cost_raw);
      //addCurvatureResidual(_params.curvature_weight, xi, xi_p1, xi_m1, curvature_params, cost_raw);
      addDistanceResidual(_params.distance_weight, xi, _original_path->at(i), cost_raw);

      if (xi[0] >= 0 && xi[0] <= _voronoi->getSizeX() && xi[1]>= 0 && xi[1] <= _voronoi->getSizeY()) {
        obs_dist = _voronoi->getDistance(xi[0], xi[1]);
        closest_obstacle = _voronoi->GetClosestObstacle(xi[0], xi[1]);
        closest_voronoi_pt = _voronoi->GetClosestVoronoiEdgePoint(xi, voronoi_dist);
        addObstacleResidual(_params.obstacle_weight, obs_dist, cost_raw);
        addVoronoiResidual(_params.voronoi_weight, _params.alpha, obs_dist, voronoi_dist, cost_raw);
        //std::cout << "Current position is: " << xi << std::endl;
        //std::cout << "Distance to the nearest obstacle is: " << obs_dist << " Point is: " << closest_obstacle << std::endl;
      }

      std::cout << "Total Cost Residual is: " << cost_raw << std::endl;

      /*********** compute gradient ***********/
      if (gradient != NULL) {
        gradient[x_index] = 0.0;
        gradient[y_index] = 0.0;    

        addSmoothingJacobian(_params.smooth_weight, xi, xi_p1, xi_m1, grad_x_raw, grad_y_raw);

        //addCurvatureJacobian(_params.curvature_weight, xi, xi_p1, xi_m1, curvature_params, grad_x_raw, grad_y_raw);

        addDistanceJacobian(_params.distance_weight, xi, _original_path->at(i), grad_x_raw, grad_y_raw);

        if (xi[0] >= 0 && xi[0] <= _voronoi->getSizeX() && xi[1]>= 0 && xi[1] <= _voronoi->getSizeY()) {
          addObstacleJacobian(_params.obstacle_weight, xi[0], xi[1], obs_dist, grad_x_raw, grad_y_raw);
          //addObstacleJacobian(_params.obstacle_weight, xi[0], xi[1], obs_dist, closest_obstacle, grad_x_raw, grad_y_raw);
          addVoronoiJacobian(_params.voronoi_weight, _params.alpha, xi, obs_dist, voronoi_dist, closest_obstacle, closest_voronoi_pt, grad_x_raw, grad_y_raw);
        }
    
        gradient[x_index] = grad_x_raw * _params.alpha / totalWeight;
        gradient[y_index] = grad_y_raw * _params.alpha / totalWeight;

        std::cout << "Total gradient is: " << grad_x_raw << "|" << grad_y_raw << "****" << std::endl;
      }

      
    }

    cost[0] = cost_raw;

    return true;
  }

  /**
   * @brief Get number of parameter blocks
   * @return Number of parameters in cost function
   */
  virtual int NumParameters() const {return _num_params;}

protected:
  /**
   * @brief Cost function term for smooth paths
   * @param weight Weight to apply to function
   * @param pt Point Xi for evaluation
   * @param pt Point Xi+1 for calculating Xi's cost
   * @param pt Point Xi-1 for calculating Xi's cost
   * @param r Residual (cost) of term
   */
  inline void addSmoothingResidual(
    const double & weight,
    const Eigen::Vector2d & pt,
    const Eigen::Vector2d & pt_p,
    const Eigen::Vector2d & pt_m,
    double & r) const
  {
    double smooth_r = weight * (
      pt_p.dot(pt_p) -
      4 * pt_p.dot(pt) +
      2 * pt_p.dot(pt_m) +
      4 * pt.dot(pt) -
      4 * pt.dot(pt_m) +
      pt_m.dot(pt_m));    // objective function value

    r += smooth_r;

    std::cout << "Smoothing residual is: " << smooth_r << std::endl;
  }

  /**
   * @brief Cost function derivative term for smooth paths
   * @param weight Weight to apply to function
   * @param pt Point Xi for evaluation
   * @param pt Point Xi+1 for calculating Xi's cost
   * @param pt Point Xi-1 for calculating Xi's cost
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  inline void addSmoothingJacobian(
    const double & weight,
    const Eigen::Vector2d & pt,
    const Eigen::Vector2d & pt_p,
    const Eigen::Vector2d & pt_m,
    double & j0,
    double & j1) const
  {
    double smoothing_jacobian_0 = weight * (-4 * pt_m[0] + 8 * pt[0] - 4 * pt_p[0]);   // xi x component of partial-derivative

    double smoothing_jacobian_1 = weight * (-4 * pt_m[1] + 8 * pt[1] - 4 * pt_p[1]);   // xi y component of partial-derivative

    j0 += smoothing_jacobian_0;
    j1 += smoothing_jacobian_1;  

    std::cout << "Smoothing jacobian is: " << smoothing_jacobian_0 << "|" << smoothing_jacobian_1 << std::endl;
  }

  /**
   * @brief Cost function term for maximum curved paths
   * @param weight Weight to apply to function
   * @param pt Point Xi for evaluation
   * @param pt Point Xi+1 for calculating Xi's cost
   * @param pt Point Xi-1 for calculating Xi's cost
   * @param curvature_params A struct to cache computations for the jacobian to use
   * @param r Residual (cost) of term
   */
  inline void addCurvatureResidual(
    const double & weight,
    const Eigen::Vector2d & pt,
    const Eigen::Vector2d & pt_p,
    const Eigen::Vector2d & pt_m,
    CurvatureComputations & curvature_params,
    double & r) const
  {
    curvature_params.valid = true;
    curvature_params.delta_xi = Eigen::Vector2d(pt[0] - pt_m[0], pt[1] - pt_m[1]);
    curvature_params.delta_xi_p = Eigen::Vector2d(pt_p[0] - pt[0], pt_p[1] - pt[1]);
    curvature_params.delta_xi_norm = curvature_params.delta_xi.norm();
    curvature_params.delta_xi_p_norm = curvature_params.delta_xi_p.norm();

    if (curvature_params.delta_xi_norm < EPSILON || curvature_params.delta_xi_p_norm < EPSILON ||
      std::isnan(curvature_params.delta_xi_p_norm) || std::isnan(curvature_params.delta_xi_norm) ||
      std::isinf(curvature_params.delta_xi_p_norm) || std::isinf(curvature_params.delta_xi_norm))
    {
      // ensure we have non-nan values returned
      curvature_params.valid = false;
      return;
    }

    const double & delta_xi_by_xi_p =
      curvature_params.delta_xi_norm * curvature_params.delta_xi_p_norm;
    double projection =
      curvature_params.delta_xi.dot(curvature_params.delta_xi_p) / delta_xi_by_xi_p;
    if (fabs(1 - projection) < EPSILON || fabs(projection + 1) < EPSILON) {
      projection = 1.0;
    }

    curvature_params.delta_phi_i = std::acos(projection);
    curvature_params.turning_rad = curvature_params.delta_phi_i / curvature_params.delta_xi_norm;

    curvature_params.ki_minus_kmax = curvature_params.turning_rad - _params.max_curvature;

    //std::cout << "*********Curvature now is: " << curvature_params.turning_rad << " Max Curvature is: " << _params.max_curvature << std::endl;

    if (curvature_params.ki_minus_kmax <= EPSILON) {
      // Quadratic penalty need not apply
      curvature_params.valid = false;
      return;
    }

    //double curvature_r = weight * curvature_params.ki_minus_kmax * curvature_params.ki_minus_kmax;  // objective function value

    double curvature_r = weight * curvature_params.ki_minus_kmax;  // objective function value

    //std::cout << "Curvature residual is: " << curvature_r << std::endl;  

    r += curvature_r;
  }

  /**
   * @brief Cost function derivative term for maximum curvature paths
   * @param weight Weight to apply to function
   * @param pt Point Xi for evaluation
   * @param pt Point Xi+1 for calculating Xi's cost
   * @param pt Point Xi-1 for calculating Xi's cost
   * @param curvature_params A struct with cached values to speed up Jacobian computation
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  inline void addCurvatureJacobian(
    const double & weight,
    const Eigen::Vector2d & pt,
    const Eigen::Vector2d & pt_p,
    const Eigen::Vector2d & /*pt_m*/,
    CurvatureComputations & curvature_params,
    double & j0,
    double & j1) const
  {
    if (!curvature_params.isValid()) {
      return;
    }

    const double & partial_delta_phi_i_wrt_cost_delta_phi_i =
      -1 / std::sqrt(1 - std::pow(std::cos(curvature_params.delta_phi_i), 2));

    auto neg_pt_plus = -1 * pt_p;
    Eigen::Vector2d p1 = normalizedOrthogonalComplement(
      pt, neg_pt_plus, curvature_params.delta_xi_norm, curvature_params.delta_xi_p_norm);
    Eigen::Vector2d p2 = normalizedOrthogonalComplement(
      neg_pt_plus, pt, curvature_params.delta_xi_p_norm, curvature_params.delta_xi_norm);

    //const double & u = 2 * curvature_params.ki_minus_kmax;

    const double & u = 1;

    const double & common_prefix =
      -(1 / curvature_params.delta_xi_norm) * partial_delta_phi_i_wrt_cost_delta_phi_i;
    const double & common_suffix = curvature_params.delta_phi_i /
      (curvature_params.delta_xi_norm * curvature_params.delta_xi_norm);

    const Eigen::Vector2d & d_delta_xi_d_xi = curvature_params.delta_xi /
      curvature_params.delta_xi_norm;

    const Eigen::Vector2d jacobian = u *
      (common_prefix * (-p1 - p2) - (common_suffix * d_delta_xi_d_xi));

    double curvature_jacobian_0 = weight * jacobian[0];  // xi x component of partial-derivative
    double curvature_jacobian_1 = weight * jacobian[1];  // xi x component of partial-derivative 

    std::cout << "Curvature jacobian is: " << curvature_jacobian_0 << "|" << curvature_jacobian_1 << std::endl;

    j0 += curvature_jacobian_0;
    j1 += curvature_jacobian_1;
  }

  /**
   * @brief Cost function derivative term for steering away changes in pose
   * @param weight Weight to apply to function
   * @param xi Point Xi for evaluation
   * @param xi_original original point Xi for evaluation
   * @param r Residual (cost) of term
   */
  inline void addDistanceResidual(
    const double & weight,
    const Eigen::Vector2d & xi,
    const Eigen::Vector2d & xi_original,
    double & r) const
  {
    double distance_r = weight * (xi - xi_original).dot(xi - xi_original);  // objective function value

    r += distance_r; 

    std::cout << "Distance residual is: " << distance_r << std::endl;  
  }

  /**
   * @brief Cost function derivative term for steering away changes in pose
   * @param weight Weight to apply to function
   * @param xi Point Xi for evaluation
   * @param xi_original original point Xi for evaluation
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  inline void addDistanceJacobian(
    const double & weight,
    const Eigen::Vector2d & xi,
    const Eigen::Vector2d & xi_original,
    double & j0,
    double & j1) const
  {
    double distance_jacobian_0 = weight * 2 * (xi[0] - xi_original[0]);  // xi y component of partial-derivative
    double distance_jacobian_1 = weight * 2 * (xi[1] - xi_original[1]);  // xi y component of partial-derivative

    j0 += distance_jacobian_0;
    j1 += distance_jacobian_1;
    
    std::cout << "Distance jacobian is: " << distance_jacobian_0 << "|" << distance_jacobian_1 << std::endl;
  }


  /**
   * @brief Cost function term for steering away from obstacles
   * @param weight Weight to apply to function
   * @param value Point Xi's cost'
   * @param params computed values to reduce overhead
   * @param r Residual (cost) of term
   */
  inline void addVoronoiResidual(
    const double & weight,
    const double & alpha, //falloff rate for voronoi field
    const double & obs,
    const double & voronoi,
    double & r) const
  {
    if (obs > Constants::vorObsDMax) {
      return;
    }

    double voronoi_r = weight * (alpha / (alpha + obs)) * (pow(obs - Constants::obsDMax, 2) / pow(Constants::obsDMax, 2)) * (voronoi / (voronoi + obs));
    
    r += voronoi_r;

    std::cout << "Voronoi residual is: " << voronoi_r << std::endl;  
  }

  /**
   * @brief Cost function derivative term for steering away from obstacles
   * @param weight Weight to apply to function
   * @param mx Point Xi's x coordinate in map frame
   * @param mx Point Xi's y coordinate in map frame
   * @param value Point Xi's cost'
   * @param params computed values to reduce overhead
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  inline void addVoronoiJacobian(
    const double & weight,
    const double & alpha, //falloff rate for voronoi field
    const Eigen::Vector2d & xi,
    const double & obs,
    const double & voronoi,
    const Eigen::Vector2d & closest_obs,
    const Eigen::Vector2d & closest_voronoi,
    double & j0,
    double & j1) const
  {
    Eigen::Vector2d gradient;
    // if (obs > Constants::obsDMax) {
    //   return;
    // } // Maybe we should take this out for voronoi

    if (obs <= Constants::vorObsDMax && obs > 1e-6) {
      if (voronoi > 0) {
        Eigen::Vector2d obsVct ((xi[0] - closest_obs[0]), (xi[1] - closest_obs[1]));
        Eigen::Vector2d edgVct ((xi[0] - closest_voronoi[0]), (xi[1] - closest_voronoi[1]));
        Eigen::Vector2d PobsDst_Pxi = obsVct / obs;
        Eigen::Vector2d PedgDst_Pxi = edgVct / voronoi;
        float PvorPtn_PedgDst = (alpha / (alpha + obs)) *
                                (pow((obs - Constants::obsDMax), 2) / pow(Constants::obsDMax, 2)) * (obs / pow((obs + voronoi), 2));

        float PvorPtn_PobsDst = (alpha / (alpha + obs)) * (voronoi / (voronoi + obs)) * ((obs - Constants::obsDMax) / pow(Constants::obsDMax, 2))
                                * (-(obs - Constants::obsDMax) / (alpha + obs) - (obs - Constants::obsDMax) / (obs + voronoi) + 2);
        gradient = weight * (PvorPtn_PobsDst * PobsDst_Pxi + PvorPtn_PedgDst * PedgDst_Pxi);

        j0 += gradient[0];
        j1 += gradient[1];
        
        std::cout << "Voronoi jacobian is: " << gradient[0] << "|" << gradient[1] << std::endl;

      }
    }
  }


  /**
  * @brief Cost function term for steering away from obstacles
  * @param weight Weight to apply to function
  * @param value Point Xi's cost'
  * @param params computed values to reduce overhead
  * @param r Residual (cost) of term
  */
  inline void addObstacleResidual(
    const double & weight,
    const double & value,
    double & r) const
  {
    //std::cout << "*****Original distance to Obstacle is: " << value << std::endl;

    if (value > Constants::obsDMax) {
      return;
    }

    double obstacle_r = weight * (value - Constants::obsDMax) * (value - Constants::obsDMax);  // objective function value

    r += obstacle_r;

    //std::cout << "Obstacle residual is: " << obstacle_r << std::endl;  
  }


  /**
   * @brief Cost function derivative term for steering away from obstacles
   * @param weight Weight to apply to function
   * @param mx Point Xi's x coordinate in map frame
   * @param mx Point Xi's y coordinate in map frame
   * @param value Point Xi's cost'
   * @param params computed values to reduce overhead
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  inline void addObstacleJacobian(
    const double & weight,
    const unsigned int & mx,
    const unsigned int & my,
    const double & value,
    double & j0,
    double & j1) const
  {
    if (value > Constants::obsDMax) {
      return;
    }

    const Eigen::Vector2d grad = getCostmapGradient(mx, my);

    const double common_prefix = 2.0 * _params.costmap_factor * weight * (value - Constants::obsDMax);

    double obstacle_jacobian_0 = common_prefix * grad[0];  // xi x component of partial-derivative
    double obstacle_jacobian_1 = common_prefix * grad[1];  // xi y component of partial-derivative

    j0 += obstacle_jacobian_0;
    j1 += obstacle_jacobian_1;

    //std::cout << "Obstacle jacobian is: " << obstacle_jacobian_0 << "|" << obstacle_jacobian_1 << std::endl;
  }


  /**
   * @brief Cost function derivative term for steering away from obstacles
   * @param weight Weight to apply to function
   * @param mx Point Xi's x coordinate in map frame
   * @param mx Point Xi's y coordinate in map frame
   * @param value Point Xi's cost'
   * @param params computed values to reduce overhead
   * @param j0 Gradient of X term
   * @param j1 Gradient of Y term
   */
  /*inline void addObstacleJacobian(
    const double & weight,
    const unsigned int & mx,
    const unsigned int & my,
    const double & value,
    const Eigen::Vector2d & closest_obs,
    double & j0,
    double & j1) const
  {
    if (value > Constants::obsDMax) {
      return;
    }
      
    Eigen::Vector2d obsVct ((mx - closest_obs[0]), (my - closest_obs[1]));
    Eigen::Vector2d gradient = 2 * weight * (value - Constants::obsDMax)*obsVct/value;

    double obstacle_jacobian_0 = gradient[0];  // xi x component of partial-derivative
    double obstacle_jacobian_1 = gradient[1];  // xi y component of partial-derivative

    j0 += obstacle_jacobian_0;
    j1 += obstacle_jacobian_1;

    //std::cout << "Obstacle jacobian is: " << obstacle_jacobian_0 << "|" << obstacle_jacobian_1 << std::endl;
  }*/


  /**
   * @brief Computing the gradient of the costmap using the 2 point numerical differentiation method
   * @param mx Point Xi's x coordinate in map frame
   * @param mx Point Xi's y coordinate in map frame
   * @param params Params reference to store gradients
   */
  inline Eigen::Vector2d getCostmapGradient(
    const unsigned int mx,
    const unsigned int my) const
  {
    // find unit vector that describes that direction
    // via 7 point taylor series approximation for gradient at Xi
    Eigen::Vector2d gradient;

    double l_1 = 0.0;
    //double l_2 = 0.0;
    //double l_3 = 0.0;
    double r_1 = 0.0;
    //double r_2 = 0.0;
    //double r_3 = 0.0;

    if (mx < _voronoi->getSizeX()) {
      r_1 = static_cast<double>(_voronoi->getDistance(mx + 1, my));
    }
    // if (mx + 1 < _voronoi->getSizeX()) {
    //   r_2 = static_cast<double>(_voronoi->getDistance(mx + 2, my));
    // }
    // if (mx + 2 < _voronoi->getSizeX()) {
    //   r_3 = static_cast<double>(_voronoi->getDistance(mx + 3, my));
    // }

    if (mx > 0) {
      l_1 = static_cast<double>(_voronoi->getDistance(mx - 1, my));
    }
    // if (mx - 1 > 0) {
    //   l_2 = static_cast<double>(_voronoi->getDistance(mx - 2, my));
    // }
    // if (mx - 2 > 0) {
    //   l_3 = static_cast<double>(_voronoi->getDistance(mx - 3, my));
    // }

    //gradient[1] = (45 * r_1 - 9 * r_2 + r_3 - 45 * l_1 + 9 * l_2 - l_3) / 60;
    gradient[0] = (r_1 - l_1) / 2;

    if (my < _voronoi->getSizeY()) {
      r_1 = static_cast<double>(_voronoi->getDistance(mx, my + 1));
    }
    // if (my + 1 < _voronoi->getSizeY()) {
    //   r_2 = static_cast<double>(_voronoi->getDistance(mx, my + 2));
    // }
    // if (my + 2 < _voronoi->getSizeY()) {
    //   r_3 = static_cast<double>(_voronoi->getDistance(mx, my + 3));
    // }

    if (my > 0) {
      l_1 = static_cast<double>(_voronoi->getDistance(mx, my - 1));
    }
    // if (my - 1 > 0) {
    //   l_2 = static_cast<double>(_voronoi->getDistance(mx, my - 2));
    // }
    // if (my - 2 > 0) {
    //   l_3 = static_cast<double>(_voronoi->getDistance(mx, my - 3));
    // }

    //gradient[0] = (45 * r_1 - 9 * r_2 + r_3 - 45 * l_1 + 9 * l_2 - l_3) / 60;
    gradient[1] = (r_1 - l_1) / 2;

    gradient.normalize();

    return gradient;
  }

  /**
   * @brief Computing the normalized orthogonal component of 2 vectors
   * @param a Vector
   * @param b Vector
   * @param norm a Vector's norm
   * @param norm b Vector's norm
   * @return Normalized vector of orthogonal components
   */
  inline Eigen::Vector2d normalizedOrthogonalComplement(
    const Eigen::Vector2d & a,
    const Eigen::Vector2d & b,
    const double & a_norm,
    const double & b_norm) const
  {
    return (a - (a.dot(b) * b / b.squaredNorm())) / (a_norm * b_norm);
  }

  std::vector<Eigen::Vector2d> * _original_path{nullptr};
  std::vector<bool> * _prims{nullptr};
  int _num_params;
  DynamicVoronoi * _voronoi{nullptr};
  HybridAStar::Constants::SmootherParams _params;
};

}  // namespace HybridAStar

#endif  // SMOOTHER_COST_FUNCTION_H
