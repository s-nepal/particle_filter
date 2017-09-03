/*
 * particle_filter.cpp
 *
 *  Created on: September 3, 2017
 *      Author: Shitoshna Nepal
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"

#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>

using namespace std;


// Set the number of particles. Initialize all particles to first position (based on estimates of 
// x, y, theta and their uncertainties from GPS) and all weights to 1. 
// Add random Gaussian noise to the position and orientation of each particle.

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	is_initialized = true;

	num_particles = 100;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {

		Particle temp;
		temp.id = i;
		temp.x = dist_x(gen);
		temp.y = dist_y(gen);
		temp.theta = dist_theta(gen);
		temp.weight = 1.0;

		weights.push_back(1.0);
		particles.push_back(temp);
	}
}


// Based on the motion data from the vehicle, move (rotate and translate) all particles

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (int i = 0;  i < num_particles; i++) {

		double theta_original = particles[i].theta;
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);

		// If the yaw rate is too small, use a different formula
		if (abs(yaw_rate) < 0.001) {

			particles[i].x += velocity * delta_t * cos(theta_original) + noise_x;
			particles[i].y += velocity * delta_t * sin(theta_original) + noise_y;
			particles[i].theta += noise_theta;

		} else {

			double theta_new = theta_original + yaw_rate * delta_t;
			particles[i].x += velocity / yaw_rate * (sin(theta_new) - sin(theta_original)) + noise_x;
			particles[i].y += velocity / yaw_rate * (cos(theta_original) - cos(theta_new)) + noise_y;
			particles[i].theta = theta_new + noise_theta;
		}
	}
}


// For each transformed co-ordinates of the particle, find the closest landmark

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations){

	// Here, the vector predicted has the list of landmarks within range of the current particle and
	// the vector observations has the list of transformed co-ordinates for that particle

	for (int i = 0; i < observations.size(); i++) {

		int closest_id;
		float min_distance = 999999;

		for (int j = 0; j < predicted.size(); j++) {

			double d_x = predicted[j].x - observations[i].x;
			double d_y = predicted[j].y - observations[i].y;
			double distance  = d_x * d_x + d_y * d_y;

			if (distance < min_distance) {
				closest_id = j;
				min_distance = distance;
			}
		}

		observations[i].id = closest_id;
	}
}


// Based on the co-ordinates of observed landmarks from the persepective of the vehicle, transform
// landmark co-ordinates to the perspective of each particle. Using the closest neighbors method,
// find the landmark in real world co-ordinates closest to each transformed co-ordinate. Finally,
// calculate the weight of each particle based on how closely the transformed co-ordinates match
// with the co-ordinates of the closest landmark
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {

	for (int  i = 0; i < num_particles; i++) {

		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;

		vector<LandmarkObs> landmarks_in_range;
		vector<LandmarkObs> transformed_coords;

		// STEP 1: Transform the co-ordinates of the landmark from the vehicle's perspective
		// to that of the current particle, i.e. pretend that the vehicle is exactly where
		// the current particle is and oriented exactly how the current particle is oriented
		// before extrapolating from there the real world co-ordinates of each of the sensed
		// landmarks.
		for (int j = 0; j < observations.size(); j++){

			int id_o = observations[j].id;
			double x_o = observations[j].x;
			double y_o = observations[j].y;

			double transformed_x = x_p + x_o * cos(theta_p) - y_o * sin(theta_p);
			double transformed_y = y_p + y_o * cos(theta_p) + x_o * sin(theta_p);

			LandmarkObs observation;
			observation.id = id_o;
			observation.x = transformed_x;
			observation.y = transformed_y;

			transformed_coords.push_back(observation);
		}

		// STEP 2: From the vantage point of the current particle, enlist landmarks that are
		// within range of the on-board sensor.
		for (int j = 0;  j < map_landmarks.landmark_list.size(); j++) {

			int id_lmark = map_landmarks.landmark_list[j].id_i;
			double x_lmark = map_landmarks.landmark_list[j].x_f;
			double y_lmark = map_landmarks.landmark_list[j].y_f;

			double d_x = x_lmark - x_p;
			double d_y = y_lmark - y_p;
			double distance = sqrt(d_x * d_x + d_y * d_y);

			if (distance < sensor_range) {

				LandmarkObs landmark_in_range;

				landmark_in_range.id = id_lmark;
				landmark_in_range.x = x_lmark;
				landmark_in_range.y = y_lmark;

				landmarks_in_range.push_back(landmark_in_range);
			}
		}

	  	// STEP 3: For each of the transformed co-ordinates, deduce which target within
	  	// range is the closest
		dataAssociation(landmarks_in_range, transformed_coords);

	   	// STEP 4: Deduce how close the transformed co-ordinate is to its closest landmark
	   	// in real world co-ordinates. The closer the match is, the higher the weight for
	   	// the particle will be.
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		
		// initialize the weight to 1.0
		double w = 1.0;

		for (int j = 0; j < transformed_coords.size(); j++){

			int id_o = transformed_coords[j].id;
			double x_o = transformed_coords[j].x;
			double y_o = transformed_coords[j].y;

			double predicted_x = landmarks_in_range[id_o].x;
			double predicted_y = landmarks_in_range[id_o].y;

			double d_x = x_o - predicted_x;
			double d_y = y_o - predicted_y;

			double a = (1 / (2*(std_x * std_x))) * d_x * d_x;
			double b = (1 / (2*(std_y * std_y))) * d_y * d_y;
			double exponential = exp(-(a + b)) / sqrt( 2.0 * M_PI * std_x * std_y);

			w *= exponential;
		}

		particles[i].weight = w;
		weights[i] = w;
	}	
}


// Resmaple particles based on their weights. The chance of survival of a particle is
// in direct proportion to its weight.
void ParticleFilter::resample(){

	vector<Particle> resampled_particles;

	default_random_engine gen;
	discrete_distribution<int> dist(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {

		int idx = dist(gen);

		Particle temp;
		temp.x = particles[idx].x;
		temp.y = particles[idx].y;
		temp.theta = particles[idx].theta;
		temp.weight = 1.0;

		resampled_particles.push_back(temp);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
