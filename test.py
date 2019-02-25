# -*- coding: utf-8 -*-
#from math import*
import math

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



Lat_A=32.060255; Lng_A=118.796877 # 南京
Lat_B=39.904211; Lng_B=116.407395 # 北京
distance=getDistance1(Lat_A,Lng_A,Lat_B,Lng_B)
print('(Lat_A, Lng_A)=({0:.6f},{1:.6f})'.format(Lat_A,Lng_A))
print('(Lat_B, Lng_B)=({0:.6f},{1:.6f})'.format(Lat_B,Lng_B))
print('Distance1={0:.3f} km'.format(distance))
print('Distance2={0:.3f} km'.format(getDistance2(Lat_A,Lng_A,Lat_B,Lng_B)))

# or  we just use 98 feet. 29.8704m  diagonal need to multiply by1.41421356237

print('45',math.degrees(math.atan(1)))
print(math.atan(25/1000))