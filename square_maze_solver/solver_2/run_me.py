import numpy as np
import cv2
import sys

## Reads image in HSV format. Accepts filepath as input argument and returns the HSV
## equivalent of the image.
def readImageHSV(filePath):
    #############  Add your Code here   ###############
    img = cv2.imread(filePath)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ###################################################
    return hsvImg

## Reads image in binary format. Accepts filepath as input argument and returns the binary
## equivalent of the image.
def readImageBinary(filePath):
    #############  Add your Code here   ###############
    img = cv2.imread(filePath, 0)
    ret, binaryImage = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    ###################################################
    return binaryImage

## The findNeighbours function takes a maze image and row and column coordinates of a cell as input arguments
## and returns a stack consisting of all the neighbours of the cell as output.
## Note :- Neighbour refers to all the adjacent cells one can traverse to from that cell provided only horizontal
## and vertical traversal is allowed.
def findNeighbours(img,row,column):
	neighbours = []
    #############  Add your Code here   ###############
	ret, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)
    
    
	cell = img[row*20:(row+1)*20, column*20:(column+1)*20]
    
	contours, hierarchy = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) == 0:
		neighbours.extend([(row, column-1), (row-1, column), (row, column+1), (row+1, column)])
	elif len(contours) > 1:
        	#array difference
        	arr=contours[0]
        	#print arr[0]
        	num = (arr[0]-arr[1])
        	num=num[0]
        	#print num
        	if num[0]<0:
            		num[0]=-num[0]
        	if num[1]<0:
            		num[1]=-num[1]
        	if num[0]<num[1]:
            		neighbours.extend([(row-1, column), (row+1, column)])
        	else:
            		neighbours.extend([(row, column-1), (row, column+1)])

    	elif len(contours) == 1:
        	peri = cv2.arcLength(contours[0], True)
        	M = cv2.moments(contours[0])
 	#	print contours[0]
		if M['m00']!=0:
			cx = int(M['m10']/M['m00'])
        		cy = int(M['m01']/M['m00'])
			centroid = cx, cy
        	else:
			centroid = (contours[0][0][0] + contours[0][1][0])/2
        #	print centroid
        	#print peri
        	if peri > 90:
            		if centroid[0] < 9:
                		neighbours.append((row, column+1))
            		elif centroid[0] > 9:
                		neighbours.append((row, column-1))
            		elif centroid[1] < 9:
                		neighbours.append((row+1, column))
            		elif centroid[1] >9:
                		neighbours.append((row-1, column))
        	if peri < 90 and peri > 50:
            		if centroid[0] < 9:
                		if centroid[1] < 9:
                    			neighbours.extend([(row, column+1), (row+1, column)])
                		else:
                    			neighbours.extend([(row-1, column), (row, column+1)])
            		else:
                		if centroid[1] < 9:
                    			neighbours.extend([(row, column-1), (row+1, column)])
                		else:
                    			neighbours.extend([(row, column-1), (row-1, column)])
                                
        	if peri < 50:
            		if centroid[0] < 9:
                		neighbours.extend([(row-1, column), (row, column+1), (row+1, column)])
            		if centroid[0] > 9:
                		neighbours.extend([(row, column-1), (row-1, column), (row+1, column)])
            		if centroid[1] < 9:
                		neighbours.extend([(row, column-1), (row, column+1), (row+1, column)])
            		if centroid[1] > 9:
                		neighbours.extend([(row, column-1), (row-1, column), (row, column+1)])
    	else:
        	print 'Error'



    ###################################################
	return neighbours

##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
##  You can change the colourCell() functions used in the previous sections to suit your requirements.

def colourCell(img,row,column,colourVal):   ## Add required arguments here.
    
    #############  Add your Code here   ###############
	dst = np.zeros((20, 20), np.uint8)
	cv2.rectangle(dst, (0,0), (20, 20), colourVal, -1)
	cv2.bitwise_and(img[row*20:(row+1)*20, (column*20):(column+1)*20], dst, img[row*20:(row+1)*20, (column*20):(column+1)*20])

    ###################################################
    	return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph(img, initial_point, listOfMarkers):  ## You can pass your own arguments in this space.
	graph = {}
    #############  Add your Code here   ###############
	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	for row in range(initial_point[1], initial_point[0]+1):
		for col in range(initial_point[1], initial_point[0]+1):
			point = (row, col)
	#		print point
			neigh = findNeighbours(img, row, col)
			graph[point] = neigh				

    ###################################################

	return graph

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.
def findPath(graph, start, end, path=[]): ## You can pass your own arguments in this space.
    #############  Add your Code here   ###############
	path = path + [start]
#	print path
        if start == end:
            return path
        if not graph.has_key(start):
	 #   print 'somethings wrong'
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = findPath(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath

    ###################################################
	return shortest     

## The findMarkers() function returns a list of coloured markers in form of a python dictionaries
## For example if a blue marker is present at (3,6) and red marker is present at (1,5) then the
## dictionary is returned as :-
##          list_of_markers = { 'Blue':(3,6), 'Red':(1,5)}

def findMarkers(img):    ## You can pass your own arguments in this space.
    list_of_markers = {}
    #############  Add your Code here   ###############
    blue_lower = np.array([110, 100, 100])
    blue_upper = np.array([130, 255, 255])
    green_lower = np.array([50, 100, 255])
    green_upper = np.array([70, 255, 255])
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    pink_lower = np.array([140, 100, 100])
    pink_upper = np.array([160, 255, 255])
    
    mask = cv2.inRange(img, blue_lower, blue_upper)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        point = contours[0][0][0]
    	marker = point[1]/20, point[0]/20
#        print marker
        list_of_markers['Blue'] = marker
    
    mask = cv2.inRange(img, green_lower, green_upper)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    if len(contours) != 0:
        point = contours[0][0][0]
        marker = point[1]/20, point[0]/20
 #       print marker
        list_of_markers['Green'] = marker    
    
    mask = cv2.inRange(img, red_lower, red_upper)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) != 0:
        point = contours[0][0][0]
        marker = point[1]/20, point[0]/20
  #      print marker
        list_of_markers['Red'] = marker
    
    mask = cv2.inRange(img, pink_lower, pink_upper)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        point = contours[0][0][0]
        marker = point[1]/20, point[0]/20
   #     print marker
        list_of_markers['Pink'] = marker

    ###################################################
    return list_of_markers

## The findOptimumPath() function returns a python list which consists of all paths that need to be traversed
## in order to start from the bottom left corner of the maze, collect all the markers by traversing to them and
## then traverse to the top right corner of the maze.

def findOptimumPath(graph, start, end, list_of_markers):     ## You can pass your own arguments in this space.
    	path_array = []
    #############  Add your Code here   ###############
	if len(list_of_markers) == 0:
		return [findPath(graph, start, end, path=[])]
	if not graph.has_key(start):
		return None
	path_array = None
	for node in list_of_markers:
		mp = findPath(graph, start, list_of_markers[node], path=[])
		dict2 = list_of_markers.copy()
		del dict2[node]
		newpath = findOptimumPath(graph, list_of_markers[node], end, dict2)
		newpath.insert(0, mp)
		if newpath:
			length = 0
			length_s = 0
			for i in range(0, len(newpath)):
				length = length + len(newpath[i])
			if path_array:
				for i in range(0, len(path_array)):
					length_s = length_s + len(path_array[i])

			if not path_array or length < length_s:
                        	path_array = newpath



    ###################################################
	return path_array
        
## The colourPath() function highlights the whole path that needs to be traversed in the maze image and
## returns the final image.

def colourPath(img, path):   ## You can pass your own arguments in this space. 
    #############  Add your Code here   ###############
	for i in path:
		for j in i:	         ## Loop to paint the solution path.
        		img = colourCell(img, j[0], j[1], 200)

    ###################################################
	return img

#####################################    Add Utility Functions Here   ###################################
##                                                                                                     ##
##                   You are free to define any functions you want in this space.                      ##
##                             The functions should be properly explained.                             ##




##                                                                                                     ##
##                                                                                                     ##
#########################################################################################################

## This is the main() function for the code, you are not allowed to change any statements in this part of
## the code. You are only allowed to change the arguments supplied in the findMarkers(), findOptimumPath()
## and colourPath() functions.

def main(filePath, flag = 0):
    imgHSV = readImageHSV(filePath)                ## Acquire HSV equivalent of image.
    listOfMarkers = findMarkers(imgHSV)              ## Acquire the list of markers with their coordinates. 
    test = str(listOfMarkers)
    imgBinary = readImageBinary(filePath)          ## Acquire the binary equivalent of image.
    initial_point = ((len(imgBinary)/20)-1,0)      ## Bottom Left Corner Cell
    final_point = (0, (len(imgBinary[0])/20) - 1)  ## Top Right Corner Cell
    pathArray = findOptimumPath(buildGraph(imgHSV, initial_point, listOfMarkers),initial_point,final_point,listOfMarkers) ## Acquire the list of paths for optimum traversal.
    print pathArray
    img = colourPath(imgBinary, pathArray)         ## Highlight the whole optimum path in the maze image
    if __name__ == "__main__":                    
        return img
    else:
        if flag == 0:
            return pathArray
        elif flag == 1:
            return test + "\n"
        else:
            return img
## Modify the filepath in this section to test your solution for different maze images.           
if __name__ == "__main__":
    filePath = sys.argv[1]                        ## Insert filepath of image here
    img = main(filePath)                 
    cv2.imshow("canvas", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


