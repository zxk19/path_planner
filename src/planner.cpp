#include "planner.h"

using namespace HybridAStar;
//###################################################
//                                        CONSTRUCTOR
//###################################################
Planner::Planner() {
  // _____
  // TODOS
  //    initializeLookups();
  //    Lookup::collisionLookup(collisionLookup);
  // ___________________
  // COLLISION DETECTION
  //    CollisionDetection configurationSpace;
  // _________________
  // TOPICS TO PUBLISH
  pubStart = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/start", 1);

  // ___________________
  // TOPICS TO SUBSCRIBE
  if (Constants::manual) {
    subMap = n.subscribe("/map", 1, &Planner::setMap, this);
  } else {
    subMap = n.subscribe("/occ_map", 1, &Planner::setMap, this);
  }

  subGoal = n.subscribe("/move_base_simple/goal", 1, &Planner::setGoal, this);
  subStart = n.subscribe("/initialpose", 1, &Planner::setStart, this);

  // __________________________
  // INITIALIZE SMOOTHER PARAMS
  Constants::OptimizerParams params;
  params.debug = true;
  smoother_params.max_curvature = 1.f/(Constants::r * 1.1); //should be 1/min_radius 0.2
  smoother_params.curvature_weight = 0.0; // 30.0;
  smoother_params.distance_weight = 0.5; //1.0; // 10.0;
  smoother_params.smooth_weight = 0.2; //0.3; // 10; //0.0 //100
  smoother_params.obstacle_weight = 0.1; //0.2;  // 5 // 25; //0.025 //100
  smoother_params.voronoi_weight = 0.3; //0.5;  // 5; //0.025 //100
  smoother_params.alpha = 0.1; //0.1
  smoother.initialize(params);
};

//###################################################
//                                       LOOKUPTABLES
//###################################################
void Planner::initializeLookups() {
  // if (Constants::dubinsLookup) {
  //   Lookup::dubinsLookup(dubinsLookup);
  // }

  Lookup::collisionLookup(collisionLookup);
}

//###################################################
//                                                MAP
//###################################################
void Planner::setMap(const nav_msgs::OccupancyGrid::Ptr map) {
  if (Constants::coutDEBUG) {
    std::cout << "I am seeing the map..." << std::endl;
  }

  grid = map;
  //update the configuration space with the current map
  configurationSpace.updateGrid(map);
  //create array for Voronoi diagram
  // ros::Time t0 = ros::Time::now();
  int height = map->info.height;
  int width = map->info.width;
  bool** binMap;
  binMap = new bool*[width];

  for (int x = 0; x < width; x++) { binMap[x] = new bool[height]; }

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      binMap[x][y] = map->data[y * width + x] ? true : false;
    }
  }

  voronoiDiagram.initializeMap(width, height, binMap);
  voronoiDiagram.update();
  voronoiDiagram.visualize();
  // ros::Time t1 = ros::Time::now();
  // ros::Duration d(t1 - t0);
  // std::cout << "created Voronoi Diagram in ms: " << d * 1000 << std::endl;

  // plan if the switch is not set to manual and a transform is available
  if (!Constants::manual && listener.canTransform("/map", ros::Time(0), "/base_link", ros::Time(0), "/map", nullptr)) {

    listener.lookupTransform("/map", "/base_link", ros::Time(0), transform);

    // assign the values to start from base_link
    start.pose.pose.position.x = transform.getOrigin().x();
    start.pose.pose.position.y = transform.getOrigin().y();
    tf::quaternionTFToMsg(transform.getRotation(), start.pose.pose.orientation);

    if (grid->info.height >= start.pose.pose.position.y && start.pose.pose.position.y >= 0 &&
        grid->info.width >= start.pose.pose.position.x && start.pose.pose.position.x >= 0) {
      // set the start as valid and plan
      validStart = true;
    } else  {
      validStart = false;
    }

    plan();
  }
}

//###################################################
//                                   INITIALIZE START
//###################################################
void Planner::setStart(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& initial) {
  float x = initial->pose.pose.position.x / Constants::cellSize;
  float y = initial->pose.pose.position.y / Constants::cellSize;
  float t = tf::getYaw(initial->pose.pose.orientation);
  // publish the start without covariance for rviz
  geometry_msgs::PoseStamped startN;
  startN.pose.position = initial->pose.pose.position;
  startN.pose.orientation = initial->pose.pose.orientation;
  startN.header.frame_id = "map";
  startN.header.stamp = ros::Time::now();

  std::cout << "I am seeing a new start x:" << x << " y:" << y << " t:" << Helper::toDeg(t) << std::endl;

  if (grid->info.height >= y && y >= 0 && grid->info.width >= x && x >= 0) {
    validStart = true;
    start = *initial;

    if (Constants::manual) { plan();}

    // publish start for RViz
    pubStart.publish(startN);
  } else {
    std::cout << "invalid start x:" << x << " y:" << y << " t:" << Helper::toDeg(t) << std::endl;
  }
}

//###################################################
//                                    INITIALIZE GOAL
//###################################################
void Planner::setGoal(const geometry_msgs::PoseStamped::ConstPtr& end) {
  // retrieving goal position
  float x = end->pose.position.x / Constants::cellSize;
  float y = end->pose.position.y / Constants::cellSize;
  float t = tf::getYaw(end->pose.orientation);

  std::cout << "I am seeing a new goal x:" << x << " y:" << y << " t:" << Helper::toDeg(t) << std::endl;

  if (grid->info.height >= y && y >= 0 && grid->info.width >= x && x >= 0) {
    validGoal = true;
    goal = *end;

    if (Constants::manual) { plan();}

  } else {
    std::cout << "invalid goal x:" << x << " y:" << y << " t:" << Helper::toDeg(t) << std::endl;
  }
}

//###################################################
//                                      PLAN THE PATH
//###################################################
void Planner::plan() {
  // if a start as well as goal are defined go ahead and plan
  if (validStart && validGoal) {

    // ___________________________
    // LISTS ALLOWCATED ROW MAJOR ORDER
    int width = grid->info.width;
    int height = grid->info.height;
    int depth = Constants::headings;
    int length = width * height * depth;
    // define list pointers and initialize lists
    Node3D* nodes3D = new Node3D[length]();
    Node2D* nodes2D = new Node2D[width * height]();

    // ________________________
    // retrieving goal position
    float x = goal.pose.position.x / Constants::cellSize;
    float y = goal.pose.position.y / Constants::cellSize;
    float t = tf::getYaw(goal.pose.orientation);
    // set theta to a value (0,2PI]
    t = Helper::normalizeHeadingRad(t);
    Node3D nGoal(x, y, t, 0, 0, nullptr);

    // _________________________
    // retrieving start position
    x = start.pose.pose.position.x / Constants::cellSize;
    y = start.pose.pose.position.y / Constants::cellSize;
    t = tf::getYaw(start.pose.pose.orientation);
    // set theta to a value (0,2PI]
    t = Helper::normalizeHeadingRad(t);
    Node3D nStart(x, y, t, 0, 0, nullptr);

    // CLEAR THE VISUALIZATION
    visualization.clear();
    // CLEAR THE PATH
    path.clear();
    smoothedPath.clear();

    // ___________________________
    // START AND TIME THE PLANNING
    ros::Time t0 = ros::Time::now();
    // FIND THE PATH    
    // nSolution is the address of an Node 3D object (goal). Also the array nodes3D contains all the expanded nodes.
    Node3D* nSolution = Algorithm::hybridAStar(nStart, nGoal, nodes3D, nodes2D, width, height, configurationSpace, visualization);
    // TRACE THE PATH, returns an array of Node3D objects, smoother.path (std::vector<Node3D>), no smoothing yet.
    smoother.tracePath(nSolution);
    ros::Time t1 = ros::Time::now();
    ros::Duration d(t1 - t0);
    std::cout << "PLANNING TIME in ms: " << d * 1000 << std::endl;

    // _______________________________________________
    // PUBLISH THE RESULTS OF THE SEARCH -- UNSMOOTHED
    // CREATE THE UNSMOOTHED PATH, returns the array of segments, nodes, and vehicle bounding box
    path.updatePath(smoother.getPath());
    path.publishPath();
    path.publishPathNodes();
    path.publishPathVehicles();
    visualization.publishNode3DCosts(nodes3D, width, height, depth);
    visualization.publishNode2DCosts(nodes2D, width, height);

    // ____________________________
    // START AND TIME THE SMOOTHING
    // t0 = ros::Time::now();
    // SMOOTHER 1
    // smoother.smoothPath(voronoiDiagram);
    // SMOOTHER 2
    // smoother.smooth(voronoiDiagram, smoother_params, configurationSpace);
    // t1 = ros::Time::now();
    // ros::Duration d_smooth(t1 - t0);
    // std::cout << "SMOOTHING TIME in ms: " << d_smooth * 1000 << std::endl;

    // _____________________________________________
    // PUBLISH THE RESULTS OF THE SEARCH -- SMOOTHED
    // CREATE THE UPDATED PATH
    smoothedPath.updatePath(smoother.getPath());
    smoothedPath.publishPath();
    smoothedPath.publishPathNodes();
    smoothedPath.publishPathVehicles();
    visualization.publishNode3DCosts(nodes3D, width, height, depth);
    visualization.publishNode2DCosts(nodes2D, width, height);

    delete [] nodes3D;
    delete [] nodes2D;

  } else {
    std::cout << "missing goal or start" << std::endl;
  }
}
