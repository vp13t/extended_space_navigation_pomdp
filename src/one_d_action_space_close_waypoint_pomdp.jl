#Load Required Packages

#using ProfileView
using POMDPs
using Distributions
using Random
import POMDPs: initialstate, actions, gen, discount, isterminal
using POMDPModels, ARDESPOT, POMDPTools
using ParticleFilters
using BenchmarkTools
using Debugger
using LinearAlgebra


#Struct definition for POMDP State
struct POMDP_state_1D_action_space
    cart:: cart_state
    pedestrians:: Array{human_state,1}
    current_path_covered_index:: Int64
end

#Struct definition for the POMDP planner
mutable struct POMDP_Planner_1D_action_space <: POMDPs.POMDP{POMDP_state_1D_action_space,Int,Array{location,1}}
    discount_factor::Float64
    pedestrian_distance_threshold::Float64
    pedestrian_collision_penalty::Float64
    goal_reward_distance_threshold::Float64
    cart_goal_reached_distance_threshold::Float64
    goal_reward::Float64
    max_cart_speed::Float64
    world::experiment_environment
    start_path_index::Int64
end

#Function to check terminal state
function is_terminal_state_pomdp_planning(s,terminal_state)
    #terminal_cart_state = location(-100,-100,-100)
    if(terminal_state.x == s.cart.x && terminal_state.y == s.cart.y)
        return true
    else
        return false
    end
end



#************************************************************************************************
#Generate different scenarios which will represent the starting belief for the POMDP planner

struct POMDP_1D_action_space_state_distribution
    world::experiment_environment
    current_belief::Array{human_probability_over_goals,1}
    start_path_index::Float64
end

function Base.rand(rng::AbstractRNG, state_distribution::POMDP_1D_action_space_state_distribution)
    pedestrians = Array{human_state,1}()
    for i in 1:length(state_distribution.world.cart_lidar_data)
        sampled_goal = Distributions.rand(rng, SparseCat(state_distribution.world.goals,state_distribution.current_belief[i].distribution))
        new_human = human_state(state_distribution.world.cart_lidar_data[i].x,state_distribution.world.cart_lidar_data[i].y,
                            state_distribution.world.cart_lidar_data[i].v,sampled_goal,
                            state_distribution.world.cart_lidar_data[i].id)
        push!(pedestrians, new_human)
    end
    return POMDP_state_1D_action_space(state_distribution.world.cart,pedestrians,state_distribution.start_path_index)
end



#************************************************************************************************
#Simulating Human One step forward in POMDP planning

function get_pedestrian_discrete_position_pomdp_planning(new_x,new_y,world)
    discretization_step_length = 1.0
    discrete_x = floor(new_x/discretization_step_length) * discretization_step_length
    discrete_y = floor(new_y/discretization_step_length) * discretization_step_length
    discrete_x = clamp(discrete_x,0,world.length)
    discrete_y = clamp(discrete_y,0,world.breadth)
    return discrete_x, discrete_y
end

function update_human_position_pomdp_planning(human, world, time_step, rng)

    rand_num = (rand(rng) - 0.5)*0.1
    #First Quadrant
    if(human.goal.x >= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Second Quadrant
    elseif(human.goal.x <= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Third Quadrant
    elseif(human.goal.x <= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Fourth Quadrant
    else(human.goal.x >= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    end

    new_x = clamp(new_x,0,world.length)
    new_y = clamp(new_y,0,world.breadth)
    #@show(new_x,new_y)
    discrete_new_x, discrete_new_y = get_pedestrian_discrete_position_pomdp_planning(new_x,new_y,world)
    #@show(discrete_new_x,discrete_new_y)
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    observed_location = location(discrete_new_x, discrete_new_y)

    return new_human_state,observed_location
end
#@code_warntype update_human_position_pomdp_planning(env.humans[2], env, 1.0, MersenneTwister(1234))



#************************************************************************************************
#Simulating the cart one step forward in POMDP planning according to its new speed

function update_cart_position_pomdp_planning(current_cart_position, new_cart_velocity, starting_index, world)
    time_interval = 1.0/new_cart_velocity
    current_x, current_y, current_theta = current_cart_position.x, current_cart_position.y, current_cart_position.theta
    length_hybrid_a_star_path = length(world.cart_hybrid_astar_path)
    cart_path = Tuple{Float64,Float64,Float64}[ (Float64(current_x), Float64(current_y), Float64(current_theta)) ]
    if(new_cart_velocity == 0.0)
        return cart_path
    else
        for i in (1:new_cart_velocity)
            #@show(starting_index, length(world.cart_hybrid_astar_path))
            steering_angle = world.cart_hybrid_astar_path[starting_index]
            if(steering_angle == 0.0)
                new_theta = current_theta
                new_x = current_x + new_cart_velocity*cos(current_theta)*time_interval
                new_y = current_y + new_cart_velocity*sin(current_theta)*time_interval
            else
                new_theta = current_theta + (new_cart_velocity * tan(steering_angle) * time_interval / world.cart.L)
                new_theta = wrap_between_0_and_2Pi(new_theta)
                new_x = current_x + ((world.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
                new_y = current_y + ((world.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
            end
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
            starting_index = starting_index + 1
            if(starting_index>length_hybrid_a_star_path)
                for j in i+1:new_cart_velocity
                    push!(cart_path,(current_x, current_y, current_theta))
                end
                return cart_path
            end
        end
    end
    return cart_path
end
# @code_warntype update_cart_position_pomdp_planning(env.cart, 0.0, 1, env)
# update_cart_position_pomdp_planning(env.cart, 0, 1, env)


#************************************************************************************************
# Reward functions for the POMDP model

# Collision Penalty
function collision_penalty_pomdp_planning(collision_flag, penalty)
    if(collision_flag)
        return penalty
    else
        return 0.0
    end
end

# Goal Reward
function goal_reward_pomdp_planning(s, distance_threshold, goal_reached_flag, goal_reward)
    if(goal_reached_flag)
        return goal_reward
    else
        euclidean_distance = ((s.cart.x - s.cart.goal.x)^2 + (s.cart.y - s.cart.goal.y)^2)^0.5
        if(euclidean_distance<distance_threshold && euclidean_distance!=0.0)
            return goal_reward/euclidean_distance
        else
            return 0.0
        end
    end
end

# Low Speed Penalty
function speed_reward_pomdp_planning(s, max_speed)
    t::Float64 = (s.cart.v - max_speed)/max_speed
    return t
end

# Penalty for excessive acceleration and deceleration
function unsmooth_motion_penalty_pomdp_planning(s,a,sp)
    if(a==1.0 || a==-1.0)
        return -1.0
    else
        return 0.0
    end
end

function immediate_stop_penalty_pomdp_planning(immediate_stop_flag, penalty)
    if(immediate_stop_flag)
        return penalty/10.0
    else
        return 0.0
    end
end


#************************************************************************************************
#POMDP Generative Model

function POMDPs.gen(m::POMDP_Planner_1D_action_space, s, a, rng)

    cart_reached_goal_flag = false
    collision_with_pedestrian_flag = false
    immediate_stop_flag = false
    #Length of one time step
    one_time_step = 1.0
    new_human_states = human_state[]
    observed_positions = location[]

    if(is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m.cart_goal_reached_distance_threshold))
        new_cart_position = (-100.0, -100.0, -100.0)
        cart_reached_goal_flag = true
        new_cart_velocity = clamp(s.cart.v + a, 0.0, m.max_cart_speed)
        push!(observed_positions, location(-25.0,-25.0))
    elseif(s.current_path_covered_index>length(m.world.cart_hybrid_astar_path))
        new_cart_position = (-100.0, -100.0, -100.0)
        cart_reached_goal_flag = true
        new_cart_velocity = clamp(s.cart.v + a, 0.0, m.max_cart_speed)
        push!(observed_positions, location(-25.0,-25.0))
    # elseif(a == -10.0)
        # new_cart_position = (-100.0, -100.0, -100.0)
        # immediate_stop_flag = true
        # new_cart_velocity = clamp(s.cart.v + a, 0.0, m.max_cart_speed)
    else
        if(a == -10.0)
            immediate_stop_flag = true
        end
        new_cart_velocity =clamp(s.cart.v + a, 0.0, m.max_cart_speed)
        cart_path::Vector{Tuple{Float64,Float64,Float64}} = update_cart_position_pomdp_planning(s.cart, new_cart_velocity, s.current_path_covered_index, m.world)
        new_cart_position = cart_path[end]

        #Check if there is a collision with any pedestrian during the cart's path.
        #Simulate all the pedestrians.
        for human in s.pedestrians
            if(s.cart.v!=0.0)
                if( find_if_two_circles_intersect(cart_path[1][1], cart_path[1][2], s.cart.L, human.x, human.y, m.pedestrian_distance_threshold) )
                    new_cart_position = (-100.0, -100.0, -100.0)
                    collision_with_pedestrian_flag = true
                    new_human_states = human_state[]
                    observed_positions = location[ location(-50.0,-50.0) ]
                    # println("Collision with this human " ,s.pedestrians[human_index] , " ", time_index )
                    # println("Cart's position is " ,cart_path[time_index] , "\nHuman's position is ", intermediate_human_location )
                    break
                end
            end
            modified_human_state,observed_location = update_human_position_pomdp_planning(human, m.world, one_time_step, rng)
            push!(new_human_states, modified_human_state)
            push!(observed_positions, observed_location)
        end


        if(!collision_with_pedestrian_flag)
            if(new_cart_velocity!=0.0)
                for time_index in 2:(Int(new_cart_velocity)+1)
                    for human_index in 1:length(s.pedestrians)
                        intermediate_human_location = get_pedestrian_intermediate_trajectory_point(s.pedestrians[human_index].x,s.pedestrians[human_index].y,
                                                                        new_human_states[human_index].x,new_human_states[human_index].y, (1/new_cart_velocity)*(time_index-1) )
                        if( find_if_two_circles_intersect(cart_path[time_index][1], cart_path[time_index][2], s.cart.L,
                                                    intermediate_human_location[1], intermediate_human_location[2], m.pedestrian_distance_threshold) )
                            new_cart_position = (-100.0, -100.0, -100.0)
                            collision_with_pedestrian_flag = true
                            new_human_states = human_state[]
                            observed_positions = location[ location(-50.0,-50.0) ]
                            break
                        end
                    end
                    if(collision_with_pedestrian_flag)
                        cart_reached_goal_flag = false
                        break
                    end
                end
            else
                #Don't do anything
            end
        end
    end

    cart_new_state = cart_state(new_cart_position[1], new_cart_position[2], new_cart_position[3], new_cart_velocity, s.cart.L, s.cart.goal)
    # Generate starting index in Hybrid A* path for the next POMPD state
    new_path_index = s.current_path_covered_index + new_cart_velocity

    # Next POMDP State
    sp = POMDP_state_1D_action_space(cart_new_state, new_human_states, new_path_index)
    # Generated Observation
    o = observed_positions

    # R(s,a): Reward for being at state s and taking action a
    #Penalize if collision
    r = collision_penalty_pomdp_planning(collision_with_pedestrian_flag, m.pedestrian_collision_penalty)
    #println("Reward from collision ", r)
    #Reward if reached goal
    r += goal_reward_pomdp_planning(s, m.goal_reward_distance_threshold, cart_reached_goal_flag, m.goal_reward)
    #println("Reward from goal ", r)
    #Penalize if going slow when it can go fast
    r += speed_reward_pomdp_planning(s, m.max_cart_speed)
    #println("Reward from max_speed ", r)
    #Penalize if had to apply sudden brakes
    r += immediate_stop_penalty_pomdp_planning(immediate_stop_flag, m.pedestrian_collision_penalty)
    #println("Reward from immediate stoping ", r)
    #To penalize unsmooth paths
    #r += unsmooth_motion_penalty_pomdp_planning(s,a,sp) +
    #Small Penalty for longer duration paths
    r = r - 1
    return (sp=sp, o=o, r=r)
end
#@code_warntype POMDPs.gen(golfcart_pomdp(), SP_POMDP_state(env.cart,env.humans,1), 1, MersenneTwister(1234))



#************************************************************************************************
#Upper bound and lower bound values for DESPOT

function is_collision_state_pomdp_planning_1D_action_space(s,m)
    if(s.cart.v == 0.0)
        return false
    else
        for human in s.pedestrians
            if( is_within_range_check_with_points(s.cart.x,s.cart.y, human.x, human.y, m.pedestrian_distance_threshold) )
                return true
            end
        end
        return false
    end
end

function time_to_goal_pomdp_planning_1D_action_space(s,m)
    if(length(m.world.cart_hybrid_astar_path) == 0)
        remaining_path_length = sqrt( (s.cart.x-s.cart.goal.x)^2 + (s.cart.y-s.cart.goal.y)^2 )
    else
        remaining_path_length = length(m.world.cart_hybrid_astar_path) - s.current_path_covered_index
    end
    return floor(remaining_path_length/m.max_cart_speed)
end

function calculate_upper_bound_value_pomdp_planning_1D_action_space(m::POMDP_Planner_1D_action_space, b)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        if (s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif (is_within_range_check_with_points(s.cart.x, s.cart.y, s.cart.goal.x, s.cart.goal.y, m.cart_goal_reached_distance_threshold))
            value_sum += w*m.goal_reward
        elseif (is_collision_state_pomdp_planning_1D_action_space(s,m))
            value_sum += w*m.pedestrian_collision_penalty
        else
            value_sum += w*((discount(m)^time_to_goal_pomdp_planning_1D_action_space(s,m))*m.goal_reward)
        end
    end
    return value_sum
end
#@code_warntype calculate_upper_bound_value(golfcart_pomdp(), initialstate(golfcart_pomdp()))

function calculate_lower_bound_policy_pomdp_planning_1D_action_space(b)
    #Implement a reactive controller for your lower bound
    action_to_be_returned = 1.0
    d_far_threshold = 6.0
    d_near_threshold = 4.0
    for (s, w) in weighted_particles(b)
        dist_to_closest_human = 200.0  #Some big number (not Inf) that is not feasible
        for human in s.pedestrians
            euclidean_distance = sqrt((s.cart.x - human.x)^2 + (s.cart.y - human.y)^2)
            if(euclidean_distance < dist_to_closest_human)
                dist_to_closest_human = euclidean_distance
            end
            if(dist_to_closest_human<d_near_threshold)
                return -1.0
            end
        end
        if(dist_to_closest_human>d_far_threshold)
            chosen_action = 1.0
        else
            chosen_action = 0.0
        end
        if(chosen_action < action_to_be_returned)
            action_to_be_returned = chosen_action
        end
    end
    return action_to_be_returned
end



#************************************************************************************************
#Functions for debugging lb>ub error

function debug_is_collision_state_pomdp_planning_1D_action_space(s,m)
    for human in s.pedestrians
        if(is_within_range(location(s.cart.x,s.cart.y),location(human.x,human.y),m.pedestrian_distance_threshold))
            println("Collision with human ", human)
            return true
        end
    end
    return false
end

function debug_golfcart_upper_bound_1D_action_space(m,b)

    lower = lbound(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space), max_depth=100),m , b)

    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        if (s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif (is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m.cart_goal_reached_distance_threshold))
            value_sum += w*m.goal_reward
        elseif (is_collision_state_pomdp_planning_1D_action_space(s,m))
            value_sum += w*m.pedestrian_collision_penalty
        else
            value_sum += w*((discount(m)^time_to_goal_pomdp_planning_1D_action_space(s,m))*m.goal_reward)
        end
    end
    #@show("*********************************************************************")
    #@show(value_sum)
    u = (value_sum)/weight_sum(b)
    if lower > u
        push!(bad, (lower,u,b))
        @show("IN DEBUG",lower,u)
    end
    return u
end
