
import numpy as np
import arcpy
import os
import math

'''
arcpy.Clip_management(
    "E:/srtm_37_03/srtm_37_03.tif","4.101 47.879 4.424 48.104",
    "E:/srtm_37_03/srtm_37_03_clip.tif", "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
'''

# Get input Raster properties
inRas = arcpy.Raster('E:/srtm_37_03/srtm_37_03_mini.tif')
lowerLeft = arcpy.Point(inRas.extent.XMin,inRas.extent.YMin)
cellSize = inRas.meanCellWidth
spatialReference = inRas.spatialReference
'''
print('lowerLeft = ',lowerLeft)
print('cellSize = ',cellSize)
print('spatialReference = ',spatialReference)
'''

# Convert Raster to numpy array
raw_arr = arcpy.RasterToNumPyArray(inRas,nodata_to_value=-1)
#print("arr =", raw_arr)
rows = raw_arr.shape[0]
columns = raw_arr.shape[1]

'''
print("rows =", rows)
print("columns =", columns)
'''

# Get Coordinates
# n is row number!!! m is column number!!!
# Return coordinates is a list contains [longitude,latitude]
def getCoordinates(n,m):
    coordinates = []
    lon = inRas.extent.XMin + m * inRas.meanCellWidth
    lat = inRas.extent.YMin + ((rows - 1 - n) * inRas.meanCellHeight)
    coordinates.append(lon)
    coordinates.append(lat)
    #print('coordinates is', coordinates)
    return coordinates

'''
#(271,389)
Lon1 = getCoordinates(100, 100)[0]
Lat1 = getCoordinates(100, 100)[1]
Lon2 = getCoordinates(100, 101)[0]
Lat2 = getCoordinates(100, 101)[1]
'''

def getDistance1(Lat_A,Lng_A,Lat_B,Lng_B): #Haversine with 2 radius
    ra=6378.140 #Equatorial radius
    rb=6356.755 #Polar radius
    flatten=(ra-rb)/ra  # oblateness of the earth
    rad_lat_A=math.radians(Lat_A)
    rad_lng_A=math.radians(Lng_A)
    rad_lat_B=math.radians(Lat_B)
    rad_lng_B=math.radians(Lng_B)
    pA=math.atan(rb/ra*math.tan(rad_lat_A))
    pB=math.atan(rb/ra*math.tan(rad_lat_B))
    xx=math.acos(math.sin(pA)*math.sin(pB)+math.cos(pA)*math.cos(pB)*math.cos(rad_lng_A-rad_lng_B))
    c1=(math.sin(xx)-xx)*(math.sin(pA)+math.sin(pB))**2/math.cos(xx/2)**2
    c2=(math.sin(xx)+xx)*(math.sin(pA)-math.sin(pB))**2/math.sin(xx/2)**2
    dr=flatten/8*(c1-c2)
    distance=ra*(xx+dr)
    return distance

def getDistance2(lat1,lng1,lat2,lng2):# Haversine
    radlat1=math.radians(lat1)
    radlat2=math.radians(lat2)
    a=radlat1-radlat2
    b=math.radians(lng1)-math.radians(lng2)
    s=2*math.asin(math.sqrt(pow(math.sin(a/2),2)+math.cos(radlat1)*math.cos(radlat2)*pow(math.sin(b/2),2)))
    #earth_radius=6378.137
    earth_radius=6371.009 # averaged radius
    s=s*earth_radius
    if s<0:
        return -s
    else:
        return s


'''
distancePixcel1 = getDistance1(Lat1,Lon1,Lat2,Lon2)
distancePixcel2 = getDistance2(Lat1,Lon1,Lat2,Lon2)
print('distancePixcel1 is ',distancePixcel1)
print('distancePixcel2 is ',distancePixcel2)
'''

# let set the distance between each point  = 29.870
# the diagonal distance is 30*1.41421356237 = 42.243


# Gradient Calculation
# n is row number m is column number
# getGradient has direction, can be negative value
def getGradient(n_a, m_a, n_b, m_b):
    coord_lat_a = getCoordinates(n_a, m_a)[1]
    coord_lon_a = getCoordinates(n_a, m_a)[0]
    coord_lat_b = getCoordinates(n_b, m_b)[1]
    coord_lon_b = getCoordinates(n_b, m_b)[0]
    height = raw_arr[n_b][m_b] - raw_arr[n_a][m_a]
    # *1000 because getdistance function returns the km unite
    distance_ab = getDistance1(coord_lat_a,coord_lon_a, coord_lat_b,coord_lon_b)*1000
    gradient = math.degrees(math.atan((height/distance_ab)))
    '''
    print('%f,%f,%f,%f'%(coord_lon_a,coord_lat_a,coord_lon_b,coord_lat_b))
    print('height = ',height)
    print('distance_ab = ',distance_ab)
    print('gradient =',gradient)
    '''
    return gradient


'''
getGradient(1,1,135,196)
'''


# parameter all start from 1
def dijkstra(startNode_n, startNode_m, endNode_n, endNode_m):
    # to match with the all array start from 0
    startNode_n = startNode_n-1
    startNode_m = startNode_m-1
    endNode_n = endNode_n-1
    endNode_m = endNode_m-1

    count = 0

    the_shortest_distance = None
    rowArr_shape = raw_arr.shape
    # This array shows the node is visited or not, the initial array
    # is all 0 which means all nodes are unvisited so far, once visited will turn to 1
    visitUnvisitArr = np.zeros(rowArr_shape, dtype = np.int)
    #if all the nodes are visited in the end the arr should be the same as this one (used for comparision and end the lope)
    allvisited_arr = np.full(rowArr_shape, 1,dtype = np.int)
    # this array store the shortest distance. the initial value is inf which should means it is infinite
    # dtype is float because np.inf is float
    shortDistArr = np.full(rowArr_shape,np.inf)

    #shortDistArr = np.zeros(rowArr_shape, dtype = np.int)
    # preVertex array stores all the preVertex this is node (for calculate the whole shortest path in the end)
    preVertex = [[[] for col in range(raw_arr.shape[1])] for row in range(raw_arr.shape[0])]

    # the distance to itself is 0.
    shortDistArr[startNode_n][startNode_m] = 0

    current_node_n = startNode_n
    current_node_m = startNode_m
    # loop until the all the nodes are visited
    while not ((visitUnvisitArr == allvisited_arr).all()):
        # update shortest distance of 9 cells around the current node
        for i in range(current_node_m-1,current_node_m+2):
            # prevent out of bound of array
            if i < 0 or i > (raw_arr.shape[1]-1):
                continue
            for j in range(current_node_n-1,current_node_n+2):
                if j < 0 or j > (raw_arr.shape[0]-1):
                    continue
                # skip itself
                elif i == current_node_m and j == current_node_n:
                    continue
                # skip the visited node
                elif visitUnvisitArr[j][i] == 1:
                    continue
                else:
                    # these are the 8 connected nodes around this cell   ( np.inf == np.inf +1 )
                    # update the shortest distance. "Relaxation step"!!!
                    if shortDistArr[j][i] > (shortDistArr[current_node_n][current_node_m] + getDistance1(getCoordinates(current_node_n,current_node_m)[1],
                                                                                                         getCoordinates(current_node_n,current_node_m)[0],
                                                                                                         getCoordinates(j,i)[1],getCoordinates(j,i)[0])):

                        shortDistArr[j][i] = (shortDistArr[current_node_n][current_node_m] + getDistance1(getCoordinates(current_node_n,current_node_m)[1],
                                                                                                          getCoordinates(current_node_n,current_node_m)[0],
                                                                                                          getCoordinates(j,i)[1],getCoordinates(j,i)[0]))

                        # put previous vertex [j,i] in to cell for later calculate the whole path
                        # some nodes are "relaxated" before so there are values already there. do not append use update!
                        if preVertex[j][i] ==[]:
                            preVertex[j][i].append(current_node_n)
                            preVertex[j][i].append(current_node_m)
                        elif preVertex[j][i] !=[]:
                            preVertex[j][i][0] = current_node_n
                            preVertex[j][i][1] = current_node_m


        # 1 means already visited and will not be visit again
        visitUnvisitArr[current_node_n][current_node_m] = 1

        temp_low = np.inf
        temp_low_n = None
        temp_low_m = None


        for q in range(shortDistArr.shape[0]):
            for w in range(shortDistArr.shape[1]):
                if visitUnvisitArr[q][w] == 1:
                    continue
                elif visitUnvisitArr[q][w] ==0:
                    if temp_low > shortDistArr[q][w]:
                        temp_low = shortDistArr[q][w]
                        temp_low_n = q
                        temp_low_m = w

        current_node_n = temp_low_n
        current_node_m = temp_low_m
        #print ("current_node_n",current_node_n)
        #print ("current_node_m",current_node_m)

    else:
            the_shortest_distance = shortDistArr[endNode_n][endNode_m]
            the_shortest_path = []
            #print(preVertex)
            pre_node_n = preVertex[endNode_n][endNode_m][0]
            pre_node_m = preVertex[endNode_n][endNode_m][1]
            the_shortest_path.insert(0,[endNode_n,endNode_m])

            #print('startNode %d%d'%(startNode_n,startNode_m))

            # iterate to the start node
            # !!!or!!!
            while pre_node_n != startNode_n or pre_node_m != startNode_m:
                the_shortest_path.insert(0,[pre_node_n,pre_node_m])
                pre_node_n = preVertex[pre_node_n][pre_node_m][0]
                pre_node_m = preVertex[pre_node_n][pre_node_m][1]
            else:
                # print(shortDistArr)
                the_shortest_path.insert(0, [startNode_n, startNode_m])
                print('the shortest distance :',the_shortest_distance)
                print('Path :',the_shortest_path)

                # this array is only for visualizing the path.
                demonstrate_array = np.full(rowArr_shape,np.nan)
                for p in range(len(the_shortest_path)):
                    demonstrate_array[the_shortest_path[p][0]][the_shortest_path[p][1]] = 0
                print("Route:",demonstrate_array)


dijkstra(1, 1, 8, 12)









'''
        # visitedNode is a tuple contains non zero cell indexes which are about to find the lowest cell
        unvisitedNodes = np.where(visitUnvisitArr == 0)

        lowest_value = np.inf
        lowest_node_n = None
        lowest_node_m = None
        # find the lowest value and the node in all the unvisited nodes
        # proceed to the next node
        for k in range(len(unvisitedNodes[0])):

            if len(unvisitedNodes[0]) <= 0:
                break
            # set the first trial the lowest then compare with others later
            if k == 0:
                lowest_value = shortDistArr[unvisitedNodes[0][k],unvisitedNodes[1][k]]
                lowest_node_n = unvisitedNodes[0][k]
                lowest_node_m = unvisitedNodes[1][k]
            elif k != 0:
                if shortDistArr[unvisitedNodes[0][k], unvisitedNodes[1][k]] < lowest_value:
                    lowest_value = shortDistArr[unvisitedNodes[0][k], unvisitedNodes[1][k]]
                    lowest_node_n = unvisitedNodes[0][k]
                    lowest_node_m = unvisitedNodes[1][k]
                    #print ('lowest_node_n =',lowest_node_n)

        if len(unvisitedNodes[0]) > 0:
            current_node_n = lowest_node_n
            current_node_m = lowest_node_m
            count = count +1
            print('current location : %d,%d'%(current_node_n,current_node_m))
            print(count)


    else:
        the_shortest_distance = shortDistArr[endNode_n][endNode_m]
        the_shortest_path = []
        print("endNoden: ",endNode_n)
        print("endNodem: ", endNode_m)
        print(preVertex)
        pre_node_n= preVertex[endNode_n][endNode_m][0]
        pre_node_m = preVertex[endNode_n][endNode_m][1]
        the_shortest_path.insert(0,[endNode_n,endNode_m])
        # iterate to the start node
        while pre_node_n != startNode_n and pre_node_m != startNode_m:
            the_shortest_path.insert(0,[pre_node_n,pre_node_m])
            pre_node_n = preVertex[pre_node_n][pre_node_m][0]
            pre_node_m = preVertex[pre_node_n][pre_node_m][1]
        else:
            print(shortDistArr)
            print(the_shortest_path)

'''













