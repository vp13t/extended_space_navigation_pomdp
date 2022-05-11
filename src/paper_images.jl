function generate_small_environment_for_paper_image()

    world_length = 10.0
    world_breadth = 10.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,7.50)
    all_goals_list = [g1,g2,g3,g4]

    o1 = obstacle_location(6.50,4.50,1.5)
    all_obstacle_list = [o1]
    # o1 = obstacle_location(50.0,50.0,20.0)
    # o2 = obstacle_location(50.0,20.0,20.0)
    # o3 = obstacle_location(80.0,50.0,20.0)
    # o4 = obstacle_location(80.0,20.0,20.0)
    # all_obstacle_list = [o1,o2,o3,o4]

    golfcart = cart_state(1.0,2.5,0.0,2.0,1.0,cart_goal)
    initial_cart_lidar_data = Array{human_state,1}()
    initial_complete_cart_lidar_data = Array{human_state,1}()

    number_of_humans = 4
    max_num_humans = number_of_humans
    human_state_start_list = Array{human_state,1}()
    human1 = human_state(4.3,3,1.0,g1,1)
    # human2 = human_state(5,9,1.0,g2,2)
    human3 = human_state(2,7,1.0,g3,3)
    human4 = human_state(7.3,2,1.0,g1,4)
    push!(human_state_start_list,human1)
    # push!(human_state_start_list,human2)
    push!(human_state_start_list,human3)
    push!(human_state_start_list,human4)
    world = experiment_environment(world_length,world_breadth,max_num_humans,number_of_humans,
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[],location(golfcart.x, golfcart.y))


    # prob_dist_tuple = [(human1,human_probability_over_goals([1.0,0.0,0.0,0.0])) , (human2,human_probability_over_goals([0.0,1.0,0.0,0.0])),
    #                         (human3,human_probability_over_goals([0.0,0.0,1.0,0.0])), (human4,human_probability_over_goals([1.0,0.0,0.0,0.0])) ]
    prob_dist_tuple = [(human3,human_probability_over_goals([0.0,0.0,1.0,0.0])), (human4,human_probability_over_goals([1.0,0.0,0.0,0.0])) ]
    world.cart_hybrid_astar_path = hybrid_a_star_search(world.cart.x, world.cart.y,world.cart.theta, world.cart.goal.x, world.cart.goal.y, world,
                                            prob_dist_tuple,100.0);

    return world
end

function get_hybrid_astar_path_points(env)
    current_x, current_y, current_theta = env.cart.x, env.cart.y, env.cart.theta
    path_x, path_y,path_theta = [env.cart.x],[env.cart.y],[env.cart.theta]
    arc_length = 1.0
    time_interval = 1.0
    num_time_intervals = 10
    for delta_angle in env.cart_hybrid_astar_path
        final_orientation_angle = wrap_between_0_and_2Pi(current_theta+delta_angle)
        for j in 1:10
            if(delta_angle == 0.0)
                new_theta = current_theta
                new_x = current_x + arc_length*cos(current_theta)*(1/num_time_intervals)
                new_y = current_y + arc_length*sin(current_theta)*(1/num_time_intervals)
            else
                new_theta = current_theta + (delta_angle * (1/num_time_intervals))
                new_theta = wrap_between_0_and_2Pi(new_theta)
                new_x = current_x + arc_length*cos(final_orientation_angle)*(1/num_time_intervals)
                new_y = current_y + arc_length*sin(final_orientation_angle)*(1/num_time_intervals)
            end
            push!(path_x,new_x)
            push!(path_y,new_y)
            push!(path_theta,new_theta)
            current_x, current_y,current_theta = new_x,new_y,new_theta
        end
    end
    # plot!(path_x,path_y,color="grey")
    return (path_x, path_y)
end

function get_new_human_position_actual_environemnt_for_paper_image(human, world, time_step, user_defined_rng)

    rand_num = (rand(user_defined_rng) - 0.5)*0
    #rand_num = 0.0
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
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    return new_human_state
end


function display_that_cool_image_part2()

    env = generate_small_environment_for_paper_image()
    possible_delta_theta = [ -2*pi/9, -pi/9, 0.0, pi/9, 2*pi/9 ]

    default(dpi=300)
    p = plot([0.0],[0.0],legend=false,grid=false, axis=nothing)
    # p = plot([0.0],[0.0],legend=false,grid=false)
    # plot!([env.length], [env.breadth],legend=false)
    plot!([env.length], [env.breadth],legend=false, xaxis=false, yaxis=false)

    #Plot Golfcart
    annotate!(env.cart.x, env.cart.y, text("V", :white, :center, 20))
    scatter!([env.cart.x], [env.cart.y], shape=:circle, color="darkgreen", msize= 0.35*plot_size*cart_size/env.length)
    quiver!([env.cart.x],[env.cart.y],quiver=([0.5*cos(env.cart.theta)],[0.5*sin(env.cart.theta)]), color="black")

    #Plot Obstacles
    for i in 1: length(env.obstacles)
        scatter!([env.obstacles[i].x], [env.obstacles[i].y],color="black",shape=:circle,msize=plot_size*env.obstacles[i].r/env.length)
    end

    #Plot all the Humans
    for i in 1: length(env.humans)
        annotate!(env.humans[i].x, env.humans[i].y, text("H", :white, :center, 20))
        human_heading_angle = get_heading_angle(env.humans[i].goal.x, env.humans[i].goal.y, env.humans[i].x, env.humans[i].y)
        quiver!([env.humans[i].x], [env.humans[i].y],quiver=([0.7*cos(human_heading_angle)],[0.7*sin(human_heading_angle)]), color="black")
        scatter!([env.humans[i].x], [env.humans[i].y],color="darkgoldenrod4",msize=0.35*plot_size/env.length)
        new_human_state = get_new_human_position_actual_environemnt_for_paper_image( env.humans[i], env, 1.0, MersenneTwister(100))
        # annotate!(new_human_state.x, new_human_state.y, text("H", :white, :center, 20))
        quiver!([new_human_state.x], [new_human_state.y],quiver=([0.7*cos(human_heading_angle)],[0.7*sin(human_heading_angle)]), color="black")
        scatter!([new_human_state.x], [new_human_state.y],color="goldenrod3",msize=0.35*plot_size/env.length)
        if(i!=1)
            new_new_human_state = get_new_human_position_actual_environemnt_for_paper_image( new_human_state, env, 1.0, MersenneTwister(100))
            # annotate!(new_new_human_state.x, new_new_human_state.y, text("H", :white, :center,20))
            quiver!([new_new_human_state.x], [new_new_human_state.y],quiver=([0.5*cos(human_heading_angle)],[0.5*sin(human_heading_angle)]), color="black")
            scatter!([new_new_human_state.x], [new_new_human_state.y],color="darkgoldenrod1",msize=0.35*plot_size/env.length)
        end
    end
    #Plot the Hybrid A* path if it exists
    if(length(env.cart_hybrid_astar_path)!=0)
        px,py = get_hybrid_astar_path_points(env)
        # plot!(px,py,color="grey")
    end

    depth_one_cart_positions = []
    bad_actions = [-pi/9, 0.0, pi/9]
    for delta_angle in possible_delta_theta
        cart_path = update_cart_position_pomdp_planning_2D_action_space(env.cart, delta_angle, 2.0, env.length, env.breadth,1.0, 10)
        new_pos = cart_path[end]
        quiver!([env.cart.x],[env.cart.y],quiver=([1.5*cos(new_pos[3])],[1.5*sin(new_pos[3])]), color="black",)
        # plot!([env.cart.x, new_pos[1]], [env.cart.y, new_pos[2]],legend=false, xaxis=false, yaxis=false, color="darkgreen")
        # annotate!(new_pos[1], new_pos[2], text("V", :white, :center, 20))
        if(delta_angle in bad_actions)
            # annotate!(new_pos[1], new_pos[2], text("V(t+1)", :black, :center, 10))
            plot!([env.cart.x, new_pos[1]], [env.cart.y, new_pos[2]],legend=false, xaxis=false, yaxis=false, color="black",lw = 3)
            scatter!([new_pos[1]], [new_pos[2]], shape=:circle, color="red", msize= 0.35*plot_size*cart_size/env.length)
            # quiver!([new_pos[1]],[new_pos[2]],quiver=([cos(new_pos[3])],[sin(new_pos[3])]), color="red")
        else
            plot!([env.cart.x, new_pos[1]], [env.cart.y, new_pos[2]],legend=false, xaxis=false, yaxis=false, color="black",lw = 3)
            # annotate!(new_pos[1], new_pos[2], text("V", :white, :center, 20))
            scatter!([new_pos[1]], [new_pos[2]], shape=:circle, color="green2", msize= 0.35*plot_size*cart_size/env.length)
            quiver!([new_pos[1]],[new_pos[2]],quiver=([0.5*cos(new_pos[3])],[0.5*sin(new_pos[3])]), color="black")
            push!(depth_one_cart_positions,new_pos)
        end
    end

    depth_two_cart_positions = []
    bad_actions = [1,2,5,6,7]
    start_bad_index = 1
    for pos in depth_one_cart_positions
        temp_cs = cart_state(pos[1], pos[2], pos[3], env.cart.v, env.cart.L, env.cart.goal)
        for delta_angle in possible_delta_theta
            cart_path = update_cart_position_pomdp_planning_2D_action_space(temp_cs, delta_angle, 2.0, env.length, env.breadth,1.0, 10)
            new_pos = cart_path[end]
            if(start_bad_index in bad_actions)
                # annotate!(new_pos[1], new_pos[2], text("V", :white, :center, 20))
                plot!([temp_cs.x, new_pos[1]], [temp_cs.y, new_pos[2]],legend=false, xaxis=false, yaxis=false, color="black",lw = 3)
                scatter!([new_pos[1]], [new_pos[2]], shape=:circle, color="red", msize= 0.35*plot_size*cart_size/env.length)
                quiver!([temp_cs.x],[temp_cs.y],quiver=([1.1*cos(pos[3]+delta_angle)],[1.1*sin(pos[3]+delta_angle)]), color="black")
                # quiver!([new_pos[1]],[new_pos[2]],quiver=([cos(new_pos[3])],[sin(new_pos[3])]), color="red")
            else
                push!(depth_two_cart_positions,new_pos)
                plot!([temp_cs.x, new_pos[1]], [temp_cs.y, new_pos[2]],legend=false, xaxis=false, yaxis=false, color="black",lw = 3)
                # annotate!(new_pos[1], new_pos[2], text("V", :white, :center, 20))
                scatter!([new_pos[1]], [new_pos[2]], shape=:circle, color="darkolivegreen2", msize= 0.35*plot_size*cart_size/env.length)
                quiver!([new_pos[1]],[new_pos[2]],quiver=([0.5*cos(new_pos[3])],[0.5*sin(new_pos[3])]), color="black")
                quiver!([temp_cs.x],[temp_cs.y],quiver=([1.1*cos(pos[3]+delta_angle)],[1.1*sin(pos[3]+delta_angle)]), color="black")
            end
            start_bad_index +=1
        end
    end



    for pos in depth_two_cart_positions
        temp_env = deepcopy(env)
        temp_env.cart = cart_state(pos[1], pos[2], pos[3], env.cart.v, env.cart.L, env.cart.goal)
        temp_env.cart_hybrid_astar_path = hybrid_a_star_search(temp_env.cart.x, temp_env.cart.y,temp_env.cart.theta, temp_env.cart.goal.x,
                                            temp_env.cart.goal.y, temp_env, Array{Tuple{human_state,human_probability_over_goals},1}(),100.0);
        px,py = get_hybrid_astar_path_points(temp_env)
        plot!(px,py,color="grey",lw = 5,ls=:dash)
    end

    # for pos in depth_one_cart_positions
    #     temp_env = deepcopy(env)
    #     temp_env.cart = cart_state(pos[1], pos[2], pos[3], env.cart.v, env.cart.L, env.cart.goal)
    #     temp_env.cart_hybrid_astar_path = hybrid_a_star_search(temp_env.cart.x, temp_env.cart.y,temp_env.cart.theta, temp_env.cart.goal.x,
    #                                         temp_env.cart.goal.y, temp_env, Array{Tuple{human_state,human_probability_over_goals},1}(),100.0);
    #     px,py = get_hybrid_astar_path_points(temp_env)
    #     plot!(px,py,color="steelblue",lw = 5)
    # end
    annotate!(env.cart.goal.x, env.cart.goal.y, text("GOAL", :saddlebrown, :right, :italic, 50))
    plot!(size=(plot_size,plot_size))
    display(p)
end
