import matplotlib.pyplot    as pyplot
import numpy                as np
import cv2 
import math
# import sys
from collections import defaultdict
# import time

class Node:
    """
    Creating node class
    """
    def __init__(self, position, cost, key, p_key):
        self.position = position
        self.cost = cost  # initially all the new nodes have infinite cost attached to them
        self.parent = None
        self.key = key
        self.p_key = p_key
        

node_book = dict()
def store_node_data(position, parent_pos ,cost):
    key = 1
    for node in node_book.values():
            if position[0] == node.position[0] and position[1] == node.position[1]:
                return node  
    p_key = 0
    for key, node in node_book.items():
        if parent_pos[0] == node.position[0] and parent_pos[1] == node.position[1]:
            p_key =  key;           

    node = Node(position,cost, key, p_key)
    node_book[node.key] = node
    key +=1
    return node

def get_node_from_dict(position):
        for node in node_book.values():
            if position[0] == node.position[0] and position[1] == node.position[1]:
                return node
        return False

def get_parent(node):
        return node_book[node.p_key]
    


def half_planes(point1,point2, image, side):

    """
    half planes method for plotting the obstacles

    Returns:
        numpyArray: returns a numpy array with updated pixel values
    """
    image_copy = image.copy()
    slope = (point2[0]-point1[0])/(point2[1]-point1[1]+(1e-6))
    for y  in range(1,400):
        intercept = point1[0] - slope*point1[1]
        for x in range(1,250):
            if side :
                if (y < ((slope*x)+intercept)):
                    image_copy[x,y]= 1.0
            else:
                if (y > ((slope*x)+intercept)):
                    image_copy[x,y]= 1.0
    return image_copy

def obstacle_offset_boundry(image, boundry_image, color):
    """
    func for creating offset boundry for obstacles

    Args:
        image (numpyArray): numpy array of pixel vlues
        boundry_image (numpy array): updated values of boundries
        color (int): color values

    Returns:
        numpyArray: Updated Numpy array
    """

    temp_image = image.copy()
    for col  in range(boundry_image.shape[1]):
      for row in range(boundry_image.shape[0]):
        if boundry_image[row,col]== 1.0:
            temp_image[row,col] = color

    return temp_image


def polygon(xpts, ypts, image):
    #Drawing the polygon obstacle using half planes
    image = image.copy()

    side_1_l = half_planes((xpts[0], ypts[0]), (xpts[1], ypts[1]), image, 0)
    side_2_u = half_planes((xpts[1], ypts[1]), (xpts[2], ypts[2]), image, 1)
    side_3_l = half_planes((xpts[2], ypts[2]), (xpts[3], ypts[3]), image, 1)
    side_4_u = half_planes((xpts[3], ypts[3]), (xpts[0], ypts[0]), image, 0)

    side_1_2 = cv2.bitwise_and(side_1_l, side_2_u)
    side_1_2_3 = cv2.bitwise_and(side_1_2, side_4_u)
    side_3_4 = cv2.bitwise_and(side_4_u, side_3_l)

    side_2_l = half_planes((xpts[1], ypts[1]), (xpts[2], ypts[2]), image, 0)
    side_3_4_2 = cv2.bitwise_and(side_3_4, side_2_l)

    result = cv2.bitwise_or(side_1_2_3, side_3_4_2)

    return result

def circle(radius, center, image):
    #Drawing a circle
    im = image.copy()

    for j  in range(im.shape[1]):
      for i in range(im.shape[0]):
        if(((j-center[0])**2+(i-center[1])**2) < radius**2):
              im[i,j]= 1.0
    return im

def hexagon(center, size, image):
    #Drawing hexagon obstacle
    image = image.copy()
    points_x = []
    points_y = []
    itr = 6
    for n in range(180,540, 60):
        Point_x = center[0] + size * np.sin(n*np.pi / 180)
        Point_y = center[1] + size * np.cos(n*np.pi / 180)
        points_x.append(Point_x)
        points_y.append(Point_y)

    side_1 = half_planes((points_x[0], points_y[0]), (points_x[1], points_y[1]), image, 0)
    side_2 = half_planes((points_x[1], points_y[1]), (points_x[2], points_y[2]), image, 0)
    side_3 = half_planes((points_x[2], points_y[2]), (points_x[3], points_y[3]), image, 0)
    side_4 = half_planes((points_x[3], points_y[3]), (points_x[4], points_y[4]), image, 1)
    side_5 = half_planes((points_x[4], points_y[4]), (points_x[5], points_y[5]), image, 1)
    side_6 = half_planes((points_x[5], points_y[5]), (points_x[0], points_y[0]), image, 1)

    image = cv2.bitwise_and(side_1, side_2)
    image = cv2.bitwise_and(image, side_3)
    image = cv2.bitwise_and(image, side_4)
    image = cv2.bitwise_and(image, side_5)
    image = cv2.bitwise_and(image, side_6)
    return image


def boundry_of_map(image):

  temp_image = image.copy()

  for col in range(temp_image.shape[1]):
    temp_image[1, col] = 1

  # right boundary
  for row in range(temp_image.shape[0]):
    temp_image[row, 399] = 1

  # top boundary
  for col in range(temp_image.shape[1]):
    temp_image[temp_image.shape[0]-2, col] = 1

  # left boundary
  for row in range(temp_image.shape[0]):
    temp_image[row, 1] = 1

  return temp_image


def draw_map(image):
    #Function to draw the map with all obstacles
    img = image.copy()

    draw_hexagon = hexagon((200,100), 40, img)
    draw_circle = circle(40, (300, 185), img)
    xpts = [36, 115, 80, 105]
    ypts = [185, 210, 180, 100]
    draw_poly = polygon(xpts, ypts, img)
    map = cv2.bitwise_or(draw_hexagon,draw_circle)
    map = cv2.bitwise_or(map, draw_poly)
    map = cv2.bitwise_or(map, boundry_of_map(image))
    map = cv2.flip(map, 0)
    # map = colouring_the_map(map, [47, 47, 211])

    x_poly_boundary = [26, 130, 90, 115]
    y_poly_boundary = [185, 220, 180, 80]

    boundry_of_circle = cv2.bitwise_xor(circle(40, (300, 185), img), circle(45, (300, 185), img) )
    boundry_of_hexagon = cv2.bitwise_xor(hexagon((200,100), 40, img), hexagon((200,100), 45, img))
    boundry_of_polygon = cv2.bitwise_xor(polygon(xpts, ypts, img), polygon(x_poly_boundary, y_poly_boundary, img))


    boundary_map_image = cv2.bitwise_or(boundry_of_circle, boundry_of_hexagon)
    boundary_map_image = cv2.bitwise_or(boundary_map_image, boundry_of_polygon)
    boundary_map_image = cv2.flip(boundary_map_image, 0)

    final_map = cv2.bitwise_xor(boundary_map_image, map)
    # new_image = obstacle_offset_boundry(image, boundary_map_image, [154, 154, 239])
    return final_map

obstacles = []

def obstacle_coordinates(image):
    #getting the obstacle coordinates from the drawn map image


    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 1:
                obstacles.append((row, col)) 

def Check_all_obstacles(coordinate):

    #Function to check if the point is in obstacles
    for x_obs,y_obs in obstacles:
        if coordinate[0] == x_obs and coordinate[1] == y_obs:
            return True
        
        return False


def idx_of_min_node(open_nodes):

    #priority queue - getting the position with lowerst cost
    min_cost = float('inf')    
    min_cost_node = []
    for node in open_nodes:
        if node.cost < min_cost:
            min_cost = node.cost
            min_cost_node.clear()
            min_cost_node.append(node)
    for index, node in enumerate(open_nodes):
        if node.position[0] == min_cost_node[0].position[0] and min_cost_node[0].position[1] == node.position[1]:
            return index


def check_if_visited(position):
    for node in node_book.values():
        if position[0][0] == node.position[0] and position[0][1] == node.position[1]:
            return True 
    return False




# def check_in_closed_queue(current_pos, closed_q):

#     #Function to check if given point is in closed queue
#     for _ , node in enumerate(closed_q):
#         if node.position == current_pos:
#             return True
#     return False

        
def get_next_move(current_coordinate, action_set, action_cost):
    # Function to get next possible moves from current node location

    next_coordinates = [ (current_coordinate[0]+move[0], current_coordinate[1]+move[1]) for move in action_set]
    next_coordinates_cost = [ [next_coordinates[index], cost] for index, cost  in enumerate(action_cost)]

    checked_coordinates = list()
    for coordinates in next_coordinates_cost:
        if not Check_all_obstacles(coordinates[0]):
            checked_coordinates.append(coordinates)

    return checked_coordinates


def djikstra(image, start_pos, goal_pos):

    #Main function to implement the dijkstra algorith
    image = image.copy()

    goal_x = goal_pos[0]
    goal_y = goal_pos[1]
    
    x = start_pos[0]
    y = start_pos[1]

    image[x,y] = 1
    image[goal_x, goal_y] = 1

    # action_set = ["move_up", "move_down", "move_left", "move_right", "move_up_right", "move_up_left", "move_down_right", "move_down_left"]
    start_node = store_node_data((x,y), 1, 0)
    
    open_queue = []
    open_queue.append(start_node)
    closed_queue = []
    # parent_node = dict()

    action_set = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)] 
    action_cost = [1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4]


    while len(open_queue) !=0:
        current_node = open_queue.pop(idx_of_min_node(open_queue))
        current_position = current_node.position

        closed_queue.append(current_node)
        # print(current_position)

        if current_position[0] == goal_pos[0] and current_position[1] == goal_pos[1]:
            print("Goal reached!")
            break

        next_moves = get_next_move(current_node.position, action_set, action_cost)
        
        for move in next_moves:           
            if not check_if_visited(move):
                new_node = store_node_data(move[0], current_node.position, move[1]+ current_node.cost)
                # new_node.parent = current_node
                # parent_node[current_node] = new_node.parent
                # new_node.cost = move[1] + current_node.cost
                
                
                # index = find_node(current_node, open_queue)
                node_exist = False
                for index, node in enumerate(open_queue):
                    print("running")
                    if new_node.position[0] == node.position[0] and new_node.position[1] == node.position[1] and new_node.cost < node.cost:
                        open_queue.pop(index)
                        open_queue.append(new_node)
                        node_exist = True

                if not node_exist:
                    open_queue.append(new_node)

        for node in closed_queue:
            image[node.position[0], node.position[1]] = 1
            

        cv2.imshow('Frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     

    cv2.destroyAllWindows()

    return track_back(), len(closed_queue)


def track_back(parent_dict, start, goal):
    path = []
    path.append(goal)

    child = get_node_from_dict(goal)
    path.append(parent_dict[goal])
    # parent = parent_dict[goal]
    while True:
        child = get_parent(child)
        path.append(child.position)
        if child.position[0] == start[0] and child.position[1] == start[1]:
            break
        
    return path[::-1]


if __name__ == "__main__":
    image = np.zeros((250, 400))
    map = draw_map(image)
    obstacle_coordinates(image)

    # start_coordinate = (210, 68)  
    # goal_coordinate =  (210, 70)
    print("Enter x y of start node with space inbetween:  ", end="")
    input_by_user_s = input().split(" ")
    start_coordinate = (int(input_by_user_s[0]), int(input_by_user_s[1]))
    

    print("Enter x y of goal node with space inbetween:  ", end="")
    input_by_user_g = input().split(" ")
    goal_coordinate = (int(input_by_user_g[0]), int(input_by_user_g[1]))
    print("Your goal coordinate",goal_coordinate)
    

    planned_path, nodes_visited = djikstra(image, start_coordinate, goal_coordinate)


    # cv2.imshow('map', map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # planned_path, node_visited = 

    # if type(planned_path) == type(list()):
    #     for x,y in planned_path:
    #         map[x,y] = 1

        # cv2.imshow('path', map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # parent = djikstra(map,start_coordinate,goal_coordinate)
    # planned_path = track_back(parent, start_coordinate, goal_coordinate)
    # print(len(planned_path))

    
