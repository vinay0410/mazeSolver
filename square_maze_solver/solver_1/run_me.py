import numpy as np
import cv2
import sys

## The readImage function takes a file path as argument and returns image in binary form.
## You can copy the code you wrote for section1.py here.
def readImage(filePath):
    #############  Add your Code here   ###############
	img = cv2.imread(filePath, 0)
	ret, binaryImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    ###################################################
    	return binaryImage

## The findNeighbours function takes a maze image and row and column coordinates of a cell as input arguments
## and returns a stack consisting of all the neighbours of the cell as output.
## Note :- Neighbour refers to all the adjacent cells one can traverse to from that cell provided only horizontal
## and vertical traversal is allowed.
## You can copy the code you wrote for section1.py here.
def findNeighbours(img,row,column):
	neighbours = []
    #############  Add your Code here   ###############
	ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    
    
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

##  colourCell function takes 4 arguments:-
##            img - input image
##            row - row coordinates of cell to be coloured
##            column - column coordinates of cell to be coloured
##            colourVal - the intensity of the colour.
##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
##  You can copy the code you wrote for section1.py here.
def colourCell(img,row,column,colourVal):
    #############  Add your Code here   ###############
	dst = np.zeros((20, 20), np.uint8)
	cv2.rectangle(dst, (0,0), (20, 20), colourVal, -1)
	cv2.bitwise_and(img[row*20:(row+1)*20, (column*20):(column+1)*20], dst, img[row*20:(row+1)*20, (column*20):(column+1)*20])



    ###################################################
	return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph(img, initial_point, final_point):  ## You can pass your own arguments in this space.
	graph = {}
    #############  Add your Code here   ###############
	for row in range(initial_point[0], final_point[0]+1):
		for col in range(initial_point[1], final_point[1]+1):
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

## This is the main function where all other functions are called. It accepts filepath
## of an image as input. You are not allowed to change any code in this function.
def main(filePath, flag = 0):                 
    img = readImage(filePath)      ## Read image with specified filepath.
    breadth = len(img)/20          ## Breadthwise number of cells
    length = len(img[0])/20           ## Lengthwise number of cells
    if length == 10:
        initial_point = (0,0)      ## Start coordinates for maze solution
        final_point = (9,9)        ## End coordinates for maze solution    
    else:
        initial_point = (0,0)
        final_point = (19,19)
    graph = buildGraph(img, initial_point, final_point)       ## Build graph from maze image. Pass arguments as required.
    shortestPath = findPath(graph, initial_point, final_point, path=[])  ## Find shortest path. Pass arguments as required.
    print shortestPath             ## Print shortest path to verify
    string = str(shortestPath) + "\n"
    for i in shortestPath:         ## Loop to paint the solution path.
        img = colourCell(img, i[0], i[1], 200)
    if __name__ == '__main__':     ## Return value for main() function.
        return img
    else:
        if flag == 0:
            return string
        else:
            return graph

## The main() function is called here. Specify the filepath of image in the space given.            
if __name__ == '__main__':
    filePath = sys.argv[1]        ## File path for test image
    img = main(filePath)           ## Main function call
    cv2.imshow('canvas', img)
    cv2.imwrite('solution.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




