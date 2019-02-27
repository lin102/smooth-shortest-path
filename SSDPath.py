
import numpy as np
import arcpy
import math

'''
arcpy.Clip_management(
    "E:/srtm_37_03/srtm_37_03.tif","4.101 47.879 4.424 48.104",
    "E:/srtm_37_03/srtm_37_03_clip.tif", "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
'''

input_dem_file_path = arcpy.GetParameterAsText(0)
output_shape_file_path = arcpy.GetParameterAsText(6) +'.shp'
input_start_n = int(arcpy.GetParameterAsText(1))
input_start_m = int(arcpy.GetParameterAsText(2))
input_end_n = int(arcpy.GetParameterAsText(3))
input_end_m = int(arcpy.GetParameterAsText(4))
input_gradient_threshold = int(arcpy.GetParameterAsText(5))

# run outside the arcmap window
#input_dem_file_path = 'E:/srtm_37_03/srtm_37_03_mini.tif'
#input_mxd_map_path = r"C:\Users\Lin\Desktop\smooth_path.mxd"
#output_shape_file_path = r"E:\TSP_shapefile\the_shortest_smooth_path.shp"


# Get input Raster properties
inRas = arcpy.Raster(input_dem_file_path)
#inRas = arcpy.Raster('E:/srtm_37_03/srtm_37_03_clip.tif')
lowerLeft = arcpy.Point(inRas.extent.XMin,inRas.extent.YMin)
cellSize = inRas.meanCellWidth
spatialReference = inRas.spatialReference

'''
print('lowerLeft = ',lowerLeft)
print('cellSize = ',cellSize)
print('spatialReference = ',spatialReference)
'''

# Convert Raster to numpy array
raw_arr = arcpy.RasterToNumPyArray(inRas, nodata_to_value=-1)
rows = raw_arr.shape[0]
columns = raw_arr.shape[1]


# Get Coordinates
# n is row number!!! m is column number!!!
# Return coordinates is a list contains [longitude,latitude]
def getCoordinates(n,m):
    coordinates = []
    # +0.5  to get the longitude and latitude from the center of each cell
    lon = inRas.extent.XMin + (m+0.5) * inRas.meanCellWidth
    lat = inRas.extent.YMin + ((rows - 1 - n +0.5) * inRas.meanCellHeight)
    coordinates.append(lon)
    coordinates.append(lat)
    #print('coordinates is', coordinates)
    return coordinates

'''
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

def getDistance2(lat1,lng1,lat2,lng2):# Haversine distance
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
'''

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
    return gradient


# parameter all start from 1
def dijkstra_gradient_threshold(startNode_n, startNode_m, endNode_n, endNode_m, gradient_threshold):
    # to match with the all array start from 0
    startNode_n = startNode_n-1
    startNode_m = startNode_m-1
    endNode_n = endNode_n-1
    endNode_m = endNode_m-1

    rowArr_shape = raw_arr.shape
    # This array shows the node is visited or not, the initial array
    # is all 0 which means all nodes are unvisited so far, once visited will turn to 1
    visitUnvisitArr = np.zeros(rowArr_shape, dtype = np.int)
    #if all the nodes are visited in the end the arr should be the same as this one (used for comparision and end the lope)
    #allvisited_arr = np.full(rowArr_shape, 1,dtype = np.int)

    # this array store the shortest distance. the initial value is inf which should means it is infinite
    # dtype is float because np.inf is float
    shortDistArr = np.full(rowArr_shape,np.inf)

    #shortDistArr = np.zeros(rowArr_shape, dtype = np.int)
    # preVertex array stores all the preVertex this is node (for calculate the whole shortest path in the end)
    preVertex = [[[] for col in range(raw_arr.shape[1])] for row in range(raw_arr.shape[0])]
    # give the previous node of the start node it self for preventing later problem
    preVertex[startNode_n][startNode_m].append(startNode_n)
    preVertex[startNode_n][startNode_m].append(startNode_m)

    # the distance to itself is 0.
    shortDistArr[startNode_n][startNode_m] = 0

    '''
    # This does not work in this gradient threshold case anymore 
    # but it works if no gradient needed and global shortest path needed
    # Because in the map there are are cells will be be visited at all due to out of gradient threshold
    # so before looping make sure that how the visited map will be look like for end the loop
    # go through all the cells
    for check_i in range(rowArr_shape[0]):
        for check_j in range(rowArr_shape[1]):
            check_neighbor = 0
            check_cannot_reached_nei = 0
            # go through all the cells around this cell
            for check_m in range(check_j - 1, check_j + 2):
                # prevent out of bound of array
                if check_m < 0 or check_m > (rowArr_shape[1]-1):

                    continue
                for check_n in range(check_i - 1, check_i + 2):
                    if check_n < 0 or check_n > (rowArr_shape[0] - 1):

                        continue
                    # skip itself
                    elif check_n == check_i and check_m == check_j:
                        continue
                    else:
                        check_neighbor = check_neighbor+1
                        if abs(getGradient(check_i,check_j,check_n,check_m)) > gradient_threshold:
                            check_cannot_reached_nei = check_cannot_reached_nei+1

            # this cell can not be reached by all the neighbor cells
            if check_neighbor == check_cannot_reached_nei:
                allvisited_arr[check_i][check_j] = 0
    print('the nodes can not be reached',allvisited_arr)
    '''

    # set the start node as current node for starting loop
    current_node_n = startNode_n
    current_node_m = startNode_m
    # loop until the all the nodes are visited
    #while not ((visitUnvisitArr == allvisited_arr).all()):

    # loop until the end node is visited or break if there are no nodes available due to current parameter
    while visitUnvisitArr[endNode_n][endNode_m] != 1:
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
                # If the gradient is too high then this node can not be reached
                elif abs(getGradient(current_node_n,current_node_m,j,i)) > gradient_threshold:
                    #print("No way ahead",getGradient(current_node_n,current_node_m,j,i))
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

        # once reached the end node this job is done
        if visitUnvisitArr[endNode_n][endNode_m] == 1:
            continue

        temp_low = np.inf
        temp_low_n = None
        temp_low_m = None

        # find the lowest value and the cell in the all unvisited nodes
        for q in range(shortDistArr.shape[0]):
            for w in range(shortDistArr.shape[1]):
                if visitUnvisitArr[q][w] == 1:
                    continue
                elif visitUnvisitArr[q][w] ==0:
                    if temp_low > shortDistArr[q][w]:
                        temp_low = shortDistArr[q][w]
                        temp_low_n = q
                        temp_low_m = w

        # if there are no more nodes is available then end the loop. say it does not work in this gradient case
        if temp_low_n is None and temp_low_m is None:
            print ('-------THERE IS NO ROUTE MEET YOUR REQUIREMENT-----')
            print ('-------HINT:GIVE HIGHER GRADIENT OR CHOSE OTHER START POINT OR END POINT-----')
            arcpy.AddError("There is no way available under the given gradient threshold, \
                          please increase the gradient threshold or change to other start and end point!")
            arcpy.AddMessage("There is no way available under the given gradient threshold, \
                          please increase the gradient threshold or change to other start and end point! ")
            break

        current_node_n = temp_low_n
        current_node_m = temp_low_m

    else:
            the_shortest_distance = shortDistArr[endNode_n][endNode_m]
            # store the shortest path
            the_shortest_path = []
            pre_node_n = preVertex[endNode_n][endNode_m][0]
            pre_node_m = preVertex[endNode_n][endNode_m][1]
            the_shortest_path.insert(0,[endNode_n,endNode_m])

            # iterate to the start node
            # !!!or!!!
            while pre_node_n != startNode_n or pre_node_m != startNode_m:
                the_shortest_path.insert(0,[pre_node_n,pre_node_m])
                pre_node_n = preVertex[pre_node_n][pre_node_m][0]
                pre_node_m = preVertex[pre_node_n][pre_node_m][1]

            else:
                the_shortest_path.insert(0, [startNode_n, startNode_m])
                print('the shortest distance :',the_shortest_distance)
                print('Path :',the_shortest_path)

                # this array is only for visualizing the path.
                demonstrate_array = np.full(rowArr_shape,np.nan)
                for p in range(len(the_shortest_path)):
                    demonstrate_array[the_shortest_path[p][0]][the_shortest_path[p][1]] = 0
                print("Route:",demonstrate_array)

                # in this path the coordinates are geographic coordinates
                geo_path = []
                # turn the numpy coordinate back to geographical coordinates
                for np_coord in the_shortest_path:
                    geo_node = []
                    geo_node.append(getCoordinates(np_coord[0], np_coord[1])[0])
                    geo_node.append(getCoordinates(np_coord[0], np_coord[1])[1])
                    geo_path.append(geo_node)
                print('geo_path',geo_path)

                # create the path (polyline) shape file
                # A list that will hold each of the Polyline objects/ in this case only one polyline
                path = []
                path.append(geo_path)
                features = []
                for feature in path:
                    # Create a Polyline object based on the array of points
                    # Append to the list of Polyline objects
                    features.append(
                        arcpy.Polyline(
                            arcpy.Array([arcpy.Point(*coords) for coords in feature]),spatialReference))

                # Persist a copy of the Polyline objects using CopyFeatures
                arcpy.CopyFeatures_management(features, output_shape_file_path)

                # put the line shape file on the map---------
                # get the map document
                #mxd = arcpy.mapping.MapDocument(input_mxd_map_path)
                mxd = arcpy.mapping.MapDocument("CURRENT")
                # get the data frame
                dataframe = arcpy.mapping.ListDataFrames(mxd, "*")[0]
                # create a new layer
                smooth_path_layer = arcpy.mapping.Layer(output_shape_file_path)
                # add the layer to the map at the bottom of the TOC in data frame 0
                arcpy.mapping.AddLayer(dataframe, smooth_path_layer, "TOP")

                # Refresh things
                arcpy.RefreshActiveView()
                arcpy.RefreshTOC()
                #del mxd, dataframe, smooth_path_layer


dijkstra_gradient_threshold(input_start_n, input_start_m, input_end_n, input_end_m, input_gradient_threshold)
















