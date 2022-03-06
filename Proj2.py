import queue
import matplotlib.pyplot    as pyplot
import numpy                as np
import cv2 
import math
from collections import defaultdict

from numpy import imag

class Node:
    """
    Creating node class
    """
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost  # initially all the new nodes have infinite cost attached to them
        self.parent = None

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


def draw_polygon(xpts, ypts, image):
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

def draw_circle(radius, center, image):
    #Drawing a circle
    im = image.copy()

    for j  in range(im.shape[1]):
      for i in range(im.shape[0]):
        if(((j-center[0])**2+(i-center[1])**2) < radius**2):
              im[i,j]= 1.0
    return im

def draw_hexagon(center, size, image):
    #Drawing hexagon obstacle
    image_copy = image.copy()
    points_x = []
    points_y = []
    itr = 6
    for n in range(180,540, 60):
        Point_x = center[0] + size * np.sin(n*np.pi / 180)
        Point_y = center[1] + size * np.cos(n*np.pi / 180)
        points_x.append(Point_x)
        points_y.append(Point_y)

    side_1 = half_planes((points_x[0], points_y[0]), (points_x[1], points_y[1]), image_copy, 0)
    side_2 = half_planes((points_x[1], points_y[1]), (points_x[2], points_y[2]), image_copy, 0)
    side_3 = half_planes((points_x[2], points_y[2]), (points_x[3], points_y[3]), image_copy, 0)
    side_4 = half_planes((points_x[3], points_y[3]), (points_x[4], points_y[4]), image_copy, 1)
    side_5 = half_planes((points_x[4], points_y[4]), (points_x[5], points_y[5]), image_copy, 1)
    side_6 = half_planes((points_x[5], points_y[5]), (points_x[0], points_y[0]), image_copy, 1)

    image_copy = cv2.bitwise_and(side_1, side_2)
    image_copy = cv2.bitwise_and(image_copy, side_3)
    image_copy = cv2.bitwise_and(image_copy, side_4)
    image_copy = cv2.bitwise_and(image_copy, side_5)
    image_copy = cv2.bitwise_and(image_copy, side_6)
    return image_copy


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

    hexagon = draw_hexagon((200,100), 40, img)
    circle = draw_circle(40, (300, 185), img)
    xpts = [36, 115, 80, 105]
    ypts = [185, 210, 180, 100]
    poly = draw_polygon(xpts, ypts, img)
    map = cv2.bitwise_or(hexagon,circle)
    map = cv2.bitwise_or(map, poly)
    map = cv2.bitwise_or(map, boundry_of_map(image))
    map = cv2.flip(map, 0)
    x_poly_boundary = [26, 130, 90, 115]
    y_poly_boundary = [185, 220, 180, 80]
    boundry_of_circle = cv2.bitwise_xor(draw_circle(40, (300, 185), img), draw_circle(45, (300, 185), img) ) # Circle boundary
    boundry_of_hexagon = cv2.bitwise_xor(draw_hexagon((200,100), 40, img), draw_hexagon((200,100), 45, img)) # hexagon Boundry
    boundry_of_polygon = cv2.bitwise_xor(draw_polygon(xpts, ypts, img), draw_polygon(x_poly_boundary, y_poly_boundary, img)) #polygon boumdry


    boundary_map_image = cv2.bitwise_or(boundry_of_circle, boundry_of_hexagon)
    boundary_map_image = cv2.bitwise_or(boundary_map_image, boundry_of_polygon)
    boundary_map_image = cv2.flip(boundary_map_image, 0)

    map = cv2.bitwise_or(boundary_map_image, map)
    return map

obstacles = []

def obstacle_coordinates(image):
    #getting the obstacle coordinates from the drawn map image


    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 1:
                obstacles.append((x, y)) 

def Check_all_obstacles(coordinate):

    #Function to check if the point is in obstacles
    for x_obs,y_obs in obstacles:
        if coordinate[0] == x_obs and coordinate[1] == y_obs:
            return True
        
        return False


def idx_of_min_node(open_nodes):

    #priority queue - getting the position with lowerst cost(greedy approach)

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

def check_node_exisitance_queue(current_pos, queue_to_check):

    #Function to check if given point is in requested queue

    for _ , node in enumerate(queue_to_check):
        if node.position == current_pos:
            return True
    return False

def find_node(current_pos, queue_to_check):

    #Function to check if given current position is in queue

    for _ , node in enumerate(queue_to_check):
        if node.position == current_pos:
            return node
    return False


    
def get_next_move(current_coordinate, action_set, action_cost):
    # Function to get next possible moves from current node location 
    #returns list of coordinates

    next_coordinates = [ (current_coordinate[0]+move[0], current_coordinate[1]+move[1]) for move in action_set]
    next_coordinates_cost = [ [next_coordinates[index], cost] for index, cost  in enumerate(action_cost)]

    checked_coordinates = list()
    for coordinates in next_coordinates_cost:
        if not Check_all_obstacles(coordinates[0]):
            checked_coordinates.append(coordinates)

    return checked_coordinates


def dijkstra(image, start_pos, goal_pos):

    #Main function to implement the dijkstra algorith
    image = image.copy()

    goal_x = goal_pos[0]
    goal_y = goal_pos[1]
    
    x = start_pos[0]
    y = start_pos[1]

    # image[x,y] = 1
    # image[goal_x, goal_y] = 1
    start_node = Node((x,y), 0)
    
    open_queue = []       # queue of open nodes
    open_queue.append(start_node) #adding current start node to the open queue
    closed_queue = []     # queue of closed nodes

    action_set = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)] 
    base_cost = [1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4]


    while open_queue:
        current_node = open_queue.pop(idx_of_min_node(open_queue))
        current_position = current_node.position

        closed_queue.append(current_node) 

        if current_position[0] == goal_x and current_position[1] == goal_y:
            closed_queue.append(goal_pos)
            print("Goal reached!")
            break

        next_moves = get_next_move(current_node.position, action_set, base_cost) #getting possible next moves from the current node position
        
        for move in next_moves:           
            if not check_node_exisitance_queue(move[0], closed_queue) and not Check_all_obstacles(current_position) and move[0][0] < 400 and move[0][1] < 250:
                new_node = Node(move[0], move[1]+ current_node.cost)
                if not check_node_exisitance_queue(move[0], open_queue):
                    new_node.parent = current_node
                    open_queue.append(new_node)
            else:
                for index, node in enumerate(open_queue):
                    if (new_node.position[0] == node.position[0] and new_node.position[1] == node.position[1]) and new_node.cost < node.cost:
                        open_queue.pop(index)
                        open_queue.append(new_node)


        for node in closed_queue:
            image[node.position[0], node.position[1]] = 1
            

        cv2.imshow('Frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return closed_queue


def track_back(closed_queue, start, goal, image):

    #Function to trace back the path using goal node and adding its parent and the its parent and so on to the list
    image = image.copy()
    path = list()
    path.append(goal)
    node = find_node(goal, closed_queue)
    parent = node.parent
    while True:
        current = parent.parent
        parent = current
        path.append(current.position)
        if current.position[0] == start[0] and current.position[1] == start[1]:
                break


    return path


if __name__ == "__main__":
    image = np.zeros((250, 400))
    map = draw_map(image)
    obstacle_coordinates(image)

    # start_coordinate = (210, 65)  
    # goal_coordinate =  (210, 70)
    

    #Taking user input for start and goal coordinates
    print("Enter x y of start node with space inbetween:  ", end="")
    input_by_user_s = input().split(" ")
    if not Check_all_obstacles((input_by_user_s[0], input_by_user_s[1])):
        start_coordinate = (int(input_by_user_s[0]), int(input_by_user_s[1]))
    else:
        print("Coordinates in Obstacle space or out of map")


    print("Enter x y of goal node with space inbetween:  ", end="")
    input_by_user_g = input().split(" ")
    if not Check_all_obstacles((input_by_user_g[0], input_by_user_g[1])):
        goal_coordinate = (int(input_by_user_g[0]), int(input_by_user_g[1]))
    else:
        print("Coordinates in Obstacle space or out of map")

    closed_queue = dijkstra(map,start_coordinate,goal_coordinate)
    planned_path = track_back( closed_queue ,start_coordinate, goal_coordinate, map)

    if type(planned_path) == type(list()):
        for x,y in planned_path:
            map[x,y] = 1

        cv2.imshow("Final planned path",map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
