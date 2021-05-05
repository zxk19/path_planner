#ifndef SMOOTHER_H
#define SMOOTHER_H

#include <cmath>
#include <vector>

#include "dynamicvoronoi.h"
#include "node3d.h"
#include "vector2d.h"
#include "helper.h"
#include "constants.h"
#include "ceres/ceres.h"
#include "Eigen/Core"
#include "smoother_cost_function.h"


#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <queue>
#include <utility>



namespace HybridAStar{
/*!
   \brief This class takes a path object and smoothes the nodes of the path.

   It also uses the Voronoi diagram as well as the configuration space.
*/
class Smoother {
 public:
  Smoother() {}

  /*!
     \brief This function takes a path consisting of nodes and attempts to iteratively smooth the same using gradient descent.

     During the different interations the following cost are being calculated
     obstacleCost
     curvatureCost
     smoothnessCost
     voronoiCost
  */
  void smoothPath(DynamicVoronoi& voronoi);

  /*!
     \brief Given a node pointer the path to the root node will be traced recursively
     \param node a 3D node, usually the goal node
     \param i a parameter for counting the number of nodes
  */
  void tracePath(const Node3D* node, int i = 0, std::vector<Node3D> path = std::vector<Node3D>());

  /// returns the path of the smoother object
  const std::vector<Node3D>& getPath() {return path;}

  /// obstacleCost - pushes the path away from obstacles
  Vector2D obstacleTerm(Vector2D xi);

  /// curvatureCost - forces a maximum curvature of 1/R along the path ensuring drivability
  Vector2D curvatureTerm(Vector2D xi0, Vector2D xi1, Vector2D xi2);

  /// smoothnessCost - attempts to spread nodes equidistantly and with the same orientation
  Vector2D smoothnessTerm(Vector2D xim2, Vector2D xim1, Vector2D xi, Vector2D xip1, Vector2D xip2);

  /// voronoiCost - trade off between path length and closeness to obstaclesg
  //   Vector2D voronoiTerm(Vector2D xi);

  /// a boolean test, whether vector is on the grid or not
  bool isOnGrid(Vector2D vec) {
    if (vec.getX() >= 0 && vec.getX() < width &&
        vec.getY() >= 0 && vec.getY() < height) {
      return true;
    }
    return false;
  }

  /**
   * @brief Initialization of the smoother
   * @param params OptimizerParam struct
   */
  void initialize(const Constants::OptimizerParams params)
  {
    _debug = params.debug;

    // General Params

    // 2 most valid options: STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT
    _options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    _options.line_search_type = ceres::WOLFE;
    _options.nonlinear_conjugate_gradient_type = ceres::POLAK_RIBIERE;
    _options.line_search_interpolation_type = ceres::CUBIC;

    _options.max_num_iterations = params.max_iterations;
    _options.max_solver_time_in_seconds = params.max_time;

    _options.function_tolerance = params.fn_tol;
    _options.gradient_tolerance = params.gradient_tol;
    _options.parameter_tolerance = params.param_tol;

    _options.min_line_search_step_size = params.advanced.min_line_search_step_size;
    _options.max_num_line_search_step_size_iterations =
      params.advanced.max_num_line_search_step_size_iterations;
    _options.line_search_sufficient_function_decrease =
      params.advanced.line_search_sufficient_function_decrease;
    _options.max_line_search_step_contraction = params.advanced.max_line_search_step_contraction;
    _options.min_line_search_step_contraction = params.advanced.min_line_search_step_contraction;
    _options.max_num_line_search_direction_restarts =
      params.advanced.max_num_line_search_direction_restarts;
    _options.line_search_sufficient_curvature_decrease =
      params.advanced.line_search_sufficient_curvature_decrease;
    _options.max_line_search_step_expansion = params.advanced.max_line_search_step_expansion;

    if (_debug) {
      _options.minimizer_progress_to_stdout = true;
    } else {
      _options.logging_type = ceres::SILENT;
    }
  }

   /**
   * @brief Smoother method
   * @param path Reference to path
   * @param costmap Pointer to minimal costmap
   * @param smoother parameters weights
   * @return If smoothing was successful
   */
  bool smooth(
    DynamicVoronoi & voronoi,
    const Constants::SmootherParams & params)
  {
    this->voronoi = voronoi;
    
    _options.max_solver_time_in_seconds = params.max_time;

    // populate our smoothing paths
    std::vector<Eigen::Vector2d> initial_path;
    double parameters[this->path.size() * 2];  // NOLINT

    for (uint i = 0; i != this->path.size(); i++) {
      parameters[2 * i] = this->path[i].getX();
      parameters[2 * i + 1] = this->path[i].getY();
      initial_path.push_back(Eigen::Vector2d(this->path[i].getX(),this->path[i].getY()));
    }

    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(new UnconstrainedSmootherCostFunction(&initial_path, voronoi, params));
    ceres::Solve(_options, problem, parameters, &summary);

    if (_debug) {
      std::cout << summary.FullReport() << '\n';
    }

    if (!summary.IsSolutionUsable() || summary.initial_cost - summary.final_cost <= 0.0) {
      return false;
    }

    for (uint i = 0; i != this->path.size(); i++) {
      this->path[i].setX(parameters[2 * i]);
      this->path[i].setY(parameters[2 * i + 1]);
      if (i != (this->path.size() - 1)){
        this->path[i].setT(std::atan2((parameters[2 * i + 3] - parameters[2 * i + 1]), (parameters[2 * i + 2] - parameters[2 * i])));
      }
    }

    return true;
  }

 private:
  /// maximum possible curvature of the non-holonomic vehicle
  float kappaMax = 1.f / (Constants::r * 1.1);
  /// maximum distance to obstacles that is penalized
  float obsDMax = Constants::minRoadWidth;
  /// maximum distance for obstacles to influence the voronoi field
  float vorObsDMax = Constants::minRoadWidth;
  /// falloff rate for the voronoi field
  float alpha = 0.1;
  /// weight for the obstacle term
  float wObstacle = 0.2;
  /// weight for the voronoi term
  float wVoronoi = 0;
  /// weight for the curvature term
  float wCurvature = 0;
  /// weight for the smoothness term
  float wSmoothness = 0.2;
  /// voronoi diagram describing the topology of the map
  DynamicVoronoi voronoi;
  /// width of the map
  int width;
  /// height of the map
  int height;
  /// path to be smoothed
  std::vector<Node3D> path;

  bool _debug;
  ceres::GradientProblemSolver::Options _options; 
};

}
#endif // SMOOTHER_H
