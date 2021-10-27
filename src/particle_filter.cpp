
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

std::default_random_engine seed;

#define diff 0.000001

/**
 * mu_x, mu_y are the real position of landmark in map
 * observe_x,y are the observation value
 * this get weight for multi-variable gaussian distribution
 */
double multi_variable_prob(double noise_x, double noise_y, double observe_x, 
                           double observe_y, double mu_x, double mu_y) {
  double gauss_norm, weight;
  gauss_norm = 1 / (2 * M_PI * noise_x * noise_y);

  double exponent = (pow(observe_x - mu_x, 2) / (2 * pow(noise_x, 2))) 
                    + (pow(observe_y - mu_y, 2) / (2 * pow(noise_y, 2)));
  
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

// x means gps x, y means gps y, theta is the 2d graph theta of car
// std use for gaussian
void ParticleFilter::init(double x, double y, double theta, double std[]) {
    //just use 50 particles first
    num_particles = 50;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);
  
    Particle particle;
    for (int i = 0; i < num_particles; ++i) {
          //generate 50 particles according to the normal_distribution of 
          // gps position
          
          particle.id = i;
          particle.x = dist_x(seed);
          particle.y = dist_y(seed);
          particle.theta = dist_theta(seed);
          particle.weight = 1.0;
 
          particles.push_back(particle);
    }
  
    is_initialized = true;
    return;
}


//this apply measurement for current particles
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  
  // the noise for x,y,theta
  double noise_x = std_pos[0];
  double noise_y = std_pos[1];
  double noise_theta = std_pos[2];
  
  // create distribution for x,y,theta
  std::normal_distribution<double> dist_x(0, noise_x);
  std::normal_distribution<double> dist_y(0, noise_y);
  std::normal_distribution<double> dist_theta(0, noise_theta);
  
  //traverse all the particles
  for (int i = 0; i < particles.size(); i++) {
    double theta = particles[i].theta;
    if(fabs(yaw_rate) > diff) {
       //go with a curve
       particles[i].x  += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
       particles[i].y  += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
       particles[i].theta += yaw_rate * delta_t;
    }
    else {
       //go straight line
       particles[i].x += velocity * delta_t * cos(theta);
       particles[i].y += velocity * delta_t * sin(theta);
    }
    // Adding Noise
    particles[i].x += dist_x(seed);
    particles[i].y += dist_y(seed);
    particles[i].theta += dist_theta(seed);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   for (int i = 0; i < observations.size(); i++) {                                                                                          
    
    observations[i].id = -1;

    double min = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {                                                                                        
      double distance;
      //call the helper function to get the distance                                                                       
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min) {                                                                                      
        min = distance;                                                             
        observations[i].id = predicted[j].id;
      }
    }
  }
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
  // observe coordinate of landmark pos x,y
  double observe_x,observe_y; 
  // the particle x,y,theta
  double particle_x,particle_y,theta;
  // the landmark observation x,y in map coordinate
  double x_map;
  double y_map;      
  // the real x,y position of landmarks
  float x_landmark;
  float y_landmark;
  
  double mu_x;
  double mu_y;
  
  int id;
  double total_weight = 0; 
  double noise_x = std_landmark[0];
  double noise_y = std_landmark[1];
  
  //check which particle should keep, which should discard
  for (int i = 0; i < particles.size(); i++) {
    particle_x = particles[i].x;
    particle_y = particles[i].y;
    theta  = particles[i].theta; 
    
    // for a particle, get the in sensor range landmarks
    vector<LandmarkObs> landmarks_in_range;
    //traverse the map
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
         id = map_landmarks.landmark_list[j].id_i; 
         x_landmark = map_landmarks.landmark_list[j].x_f; 
         y_landmark = map_landmarks.landmark_list[j].y_f; 
         //only put within sensor range landmarks into vector
         if((fabs(particle_x - x_landmark) <= sensor_range) 
            && (fabs(particle_y - y_landmark) <= sensor_range)) {
               landmarks_in_range.push_back(LandmarkObs{id, x_landmark, y_landmark});
         }
    }
    
    // from vehicle observation to map     
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); j++) {
         id = observations[j].id;
         observe_x = observations[j].x;
         observe_y = observations[j].y;
      
         x_map = particle_x + (cos(theta) * observe_x) - (sin(theta) * observe_y);
         y_map = particle_y + (sin(theta) * observe_x) + (cos(theta) * observe_y);
         transformed_observations.push_back(LandmarkObs{id, x_map, y_map}); 
      
    }
    
    
    dataAssociation(landmarks_in_range, transformed_observations);
    
    // update the Weight with multi variable gaussian distribution
    particles[i].weight = 1.0; 
    for (int j = 0; j < transformed_observations.size(); j++){
      observe_x = transformed_observations[j].x;
      observe_y = transformed_observations[j].y;
      id    = transformed_observations[j].id;
      for (int k = 0; k < landmarks_in_range.size(); k++) {
        if (landmarks_in_range[k].id == id) {
            mu_x = landmarks_in_range[k].x;
            mu_y = landmarks_in_range[k].y;
            break;
        }
      }
      
      particles[i].weight *= multi_variable_prob(noise_x, noise_y, observe_x, observe_y, mu_x, mu_y);
      
    }  
    total_weight += particles[i].weight;
  }
  // Normalize the weight
  for (int i = 0; i < particles.size(); i++) {
      particles[i].weight /= total_weight;
  }   
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<double> weights;
  for (int i = 0; i < particles.size(); i++){
     weights.push_back(particles[i].weight);
  }
  
  double weight_max = *max_element(weights.begin(), weights.end());
  
  vector<Particle> new_particles_sampled;
  int len = particles.size();
  //select the random index distribution
  std::uniform_int_distribution<int> index_dist(0,len-1);
  std::uniform_real_distribution<double> beta_dist(0.0, weight_max);
  
  int index = index_dist(seed);
  double beta = 0.0;  
  //filter out the particles with less weight
  for (int i = 0; i < len; i++) {

    beta += beta_dist(seed) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % len;
    }
    new_particles_sampled.push_back(particles[index]);
  } 
  particles = new_particles_sampled;
  
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}