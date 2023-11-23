import streamlit as st
import numpy as np
import pandas as pd
import pulp
import itertools
import folium
import openrouteservice
from branca.element import Figure
from openrouteservice import convert
import sys
import time
import urllib.parse

class google_location:
    def __init__(self,key,geo,method_num):
        self.move_method = ["foot-walking","driving-car"]
        self.num = method_num
        self.client = openrouteservice.Client(key=key)
        self.geo = geo
        self.location_time = google_location.get_loc(self)
        self.transfer_time = google_location.get_time(self)
        
        
    def get_loc(self):
        geo_list = []
        for chimei in range(int(len(self.geo)/2)):
            url_data = self.geo[2*chimei]
            time_data = self.geo[2*chimei+1].split("\n")[0]
            split_url = url_data.split("/")
            zahyo = split_url[6].split(",")
            geo_list.append([urllib.parse.unquote(split_url[5]),float(zahyo[0].split("@")[1]),float(zahyo[1]),time_data])
        location_time = pd.DataFrame(geo_list, columns=["地名","latitude","longitude","stay_time"])
        return location_time

    def get_time(self):
        
        datalist = []
        line = self.location_time.to_numpy().tolist()
        for i in range(len(line)):
            p1 = float(line[i][1]), float(line[i][2])
            for j in range(i,len(line)-1):
                p2 = float(line[j+1][1]), float(line[j+1][2])
                p1r = tuple(reversed(p1))
                p2r = tuple(reversed(p2))

                # 経路計算 (Directions V2)
                
                routedict = self.client.directions((p1r, p2r),profile=self.move_method[self.num])
                datalist.append([i,j+1,float(routedict["routes"][0]["summary"]["duration"])])
        transfer_time = pd.DataFrame(datalist, columns=["出発地点","行先","移動時間[s]"])
        
        return transfer_time


class Optimization:
    def __init__(self,key,geo,lim_time,method_num):
        self.move_method = ["foot-walking","driving-car"]
        self.num = method_num
        self.client = openrouteservice.Client(key=key)
        self.geo = geo
        self.location_time = google_location.get_loc(self)
        self.transfer_time = google_location.get_time(self)
        self.lim_time_capacity = lim_time
        self.xijk,self.lim_day_count = Optimization.opt_scd(self)
        self.customer_count = len(self.location_time)
        #self.schedule = Optimization.schedule(self)
        
    def opt_scd(self):
        num_places_time = len(self.transfer_time)
        customer_count = len(self.location_time) #場所の数（id=0はdepot）
        lim_day_count = 100 #何日かlim

        loc_loc_time = self.transfer_time.to_numpy().tolist()
        point = self.location_time.to_numpy().tolist()

        cost = [[0 for i in range(customer_count)] for j in range(customer_count)]
        for i in range(num_places_time):
            cost[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(loc_loc_time[i][2])
            cost[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(loc_loc_time[i][2])
        cost = np.array(cost)

        visit = [[0 for i in range(customer_count)] for j in range(customer_count)]
        for i in range(num_places_time):
            visit[int(loc_loc_time[i][0])][int(loc_loc_time[i][1])] = float(point[int(loc_loc_time[i][1])][3])*60
            visit[int(loc_loc_time[i][1])][int(loc_loc_time[i][0])] = float(point[int(loc_loc_time[i][0])][3])*60
        visit = np.array(visit)



        for lim_day_count in range(lim_day_count+1):
            opt_TripY = pulp.LpProblem("CVRP", pulp.LpMinimize)
            
            #変数定義
            
            X_ijk = [[[pulp.LpVariable("X%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(lim_day_count)]
                    for j in range(customer_count)] for i in range(customer_count)]
            Y_ijk = [[[pulp.LpVariable("Y%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(lim_day_count)]
                    for j in range(customer_count)] for i in range(customer_count)]

            #制約条件

            for j in range(1, customer_count):
                opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i != j else 0 for i in range(customer_count) for k in range(lim_day_count)) == 1 

            for k in range(lim_day_count):
                opt_TripY += pulp.lpSum(X_ijk[0][j][k] for j in range(1,customer_count)) == 1
                opt_TripY += pulp.lpSum(X_ijk[i][0][k] for i in range(1,customer_count)) == 1

            for k in range(lim_day_count):
                for j in range(customer_count):
                    opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i != j else 0 for i in range(customer_count)) -  pulp.lpSum(X_ijk[j][i][k] for i in range(customer_count)) == 0
            for k in range(lim_day_count):
                opt_TripY += pulp.lpSum(visit[i][j] * X_ijk[i][j][k] + cost[i][j] * X_ijk[i][j][k] if i != j else 0 for i in range(customer_count) for j in range (customer_count)) <= self.lim_time_capacity
                
            #目的関数
            
            opt_TripY += pulp.lpSum(visit[i][j] * X_ijk[i][j][k] + cost[i][j] * X_ijk[i][j][k] if i != j else 0 for k in range(lim_day_count) for j in range(customer_count) for i in range (customer_count))
            
            #部分巡回路除去制約
            
            subtours = []
            for i in range(2,customer_count):
                subtours += itertools.combinations(range(1,customer_count), i)
            for s in subtours:
                opt_TripY += pulp.lpSum(X_ijk[i][j][k] if i !=j else 0 for i, j in itertools.permutations(s,2) for k in range(lim_day_count)) <= len(s) - 1
                
            if opt_TripY.solve() == 1:
                time_start = time.time()
                status = opt_TripY.solve()
                time_stop = time.time()
                print(f'ステータス:{pulp.LpStatus[status]}')
                print(f"移動方法:{self.move_method[self.num]}")
                print('日数:', lim_day_count)
                print('目的関数値:',pulp.value(opt_TripY.objective))
                print('使用時間:',f"{int(pulp.value(opt_TripY.objective)//3600)}時間{int(pulp.value(opt_TripY.objective)%3600//60)}分{pulp.value(opt_TripY.objective)%3600%60:.3}秒")
                print(f'計算時間:{(time_stop - time_start):.3}(秒)')
                break
        if not(opt_TripY.solve()) == 1:
            print("日数が足りません. プランを立て直してください.")
            sys.exit()
        
        return X_ijk,lim_day_count
    
    def schedule(self):
        basyo_num_list = []
        hiduke_judg_list = []

        for k in range(self.lim_day_count):
            for i in range(self.customer_count):
                for j in range(self.customer_count):
                    if i != j and pulp.value(self.xijk[i][j][k]) == 1:
                        #print("日付：",k)
                        #print("地点：",i)
                        #print("目的：",j,"\n")
                        basyo_num_list.append(i)
                        hiduke_judg_list.append(k)
        number = 0
        plan_list = []
        ittan_list = []
        for day in range(len(hiduke_judg_list)):
            if basyo_num_list[day] == 0:
                if number > 0:
                    plan_list.append(ittan_list)
                    ittan_list = []
                ittan_list.append("{}日目の予定".format(number+1))
                print("\n{}日目の予定".format(number+1))
                number += 1
            else:
                ittan_list.append(self.location_time.iloc[basyo_num_list[day],0])
                print(self.location_time.iloc[basyo_num_list[day],0])
        plan_list.append(ittan_list)
        a = plan_list
        return a
    

class VisualizationMap:
    def __init__(self,key,geo,lim_time,method_num):
        self.move_method = ["foot-walking","driving-car"]
        self.num = method_num
        self.client = openrouteservice.Client(key=key)
        self.geo = geo
        self.location_time = google_location.get_loc(self)
        self.transfer_time = google_location.get_time(self)
        self.lim_time_capacity = lim_time
        self.xijk,self.lim_day_count = Optimization.opt_scd(self)
        self.customer_count = len(self.location_time)
        self.schedule = Optimization.schedule(self)
        
    def map(self):
        color_list = ["red","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue","lightgreen","gray","black","lightgray","blue"]

        points_a = []
        for i in range(len(self.location_time)):
            points_a.append([self.location_time.iloc[i,1],self.location_time.iloc[i,2]])

        def route_view(points_a):
            loc_place = []
            for chimei in range(len(points_a)-1):
                p1 = points_a[chimei]
                p2 = points_a[chimei+1]
                p1r = tuple(reversed(p1))
                p2r = tuple(reversed(p2))

                # 経路計算 (Directions V2)
                
                
                routedict = self.client.directions((p1r, p2r),profile=self.move_method[self.num])
                geom = routedict["routes"][0]["geometry"]
                decoded = convert.decode_polyline(geom)
                for i in range(len(decoded["coordinates"])):
                    loc_place.append(decoded["coordinates"][i])
            return loc_place

        def reverse_lat_long(list_of_lat_long):
            return [(p[1], p[0]) for p in list_of_lat_long]


        ave_lat = sum(p[0] for p in points_a)/len(points_a)
        ave_lon = sum(p[1] for p in points_a)/len(points_a)
        fig = Figure(width=800, height=400)

        my_map = folium.Map(
            location=[ave_lat, ave_lon], 
            zoom_start=12
        )

        basyo_num_list = []
        hiduke_judg_list = []

        for k in range(self.lim_day_count):
            for i in range(self.customer_count):
                for j in range(self.customer_count):
                    if i != j and pulp.value(self.xijk[i][j][k]) == 1:
                        #print("日付：",k)
                        #print("地点：",i)
                        #print("目的：",j,"\n")
                        basyo_num_list.append(i)
                        hiduke_judg_list.append(k)
                        
        day_trip_zahyo = []
        hiduke_hantei = 0
        bangou_1 = 0

        for aaaa in hiduke_judg_list:
            bangou_2 = basyo_num_list[bangou_1]
            if not(aaaa==hiduke_hantei):
                day_trip_zahyo.append([self.location_time.iloc[0,1],self.location_time.iloc[0,2]])
                def_routeview = route_view(day_trip_zahyo)
                coord = reverse_lat_long(def_routeview)
                folium.vector_layers.PolyLine(coord,
                                                color=color_list[aaaa], 
                                                weight=2.5, 
                                                opacity=1
                                                ).add_to(my_map)
                for each in range(len(day_trip_zahyo)-2):
                    folium.Marker(
                            location=day_trip_zahyo[each+1],
                            icon = folium.Icon(color=color_list[aaaa])
                        ).add_to(my_map)
                day_trip_zahyo = []
                day_trip_zahyo.append([self.location_time.iloc[bangou_2,1],self.location_time.iloc[bangou_2,2]])
                hiduke_hantei += 1
            else:
                day_trip_zahyo.append([self.location_time.iloc[bangou_2,1],self.location_time.iloc[bangou_2,2]])
            bangou_1 += 1
            
        #最終日ルート
        day_trip_zahyo.append([self.location_time.iloc[0,1],self.location_time.iloc[0,2]])
        def_routeview = route_view(day_trip_zahyo)
        coord = reverse_lat_long(def_routeview)
        folium.vector_layers.PolyLine(coord,
                                        color=color_list[0], 
                                        weight=2.5, 
                                        opacity=1
                                        ).add_to(my_map)
        for each in range(len(day_trip_zahyo)-2):
            folium.Marker(
                    location=day_trip_zahyo[each+1],
                    icon = folium.Icon(color=color_list[0])
                ).add_to(my_map)
        folium.Marker(
            location=[self.location_time.iloc[0,1],self.location_time.iloc[0,2]],
            popup=self.location_time.iloc[0,0]
        ).add_to(my_map)

        
        return my_map


from PIL import Image

image = Image.open('logo.png')
st.set_page_config(
    page_title="TripOpt_TripY", 
    page_icon=image,
    initial_sidebar_state="expanded"
    )


st.text("TripOpt_TripY")
st.title("旅行の予定を OpenRouteService を用いて Python から最適化する")
OSM_key = st.sidebar.text_input("OpenRouteService API")
time_day = st.sidebar.slider("Time Limit(Hours)", 0, 24, 8)*60*60
method_num = st.sidebar.radio("Transportation", ("🚶：Foot-Walking", "🚙：Driving-Car"))
check = st.sidebar.multiselect("Visualization", ("Schedule", "Route-Map"))
pos_text = st.sidebar.file_uploader("Choose file", type='txt')
if method_num == "🚶：Foot-Walking":
    method_num = 0
else:
    method_num = 1
if pos_text is not None:
    data = pd.read_csv(pos_text,sep="  ",header=None)
    pos_text = list(itertools.chain.from_iterable(data.to_numpy().tolist()))
if st.sidebar.button("Optimization"):
    if (pos_text is not None) and (OSM_key is not None) and check:
        if "Schedule" in check:
            st.header("Schedule")
            a = Optimization(geo=pos_text,key=OSM_key,lim_time=time_day,method_num=method_num).schedule()
            st.write(pd.DataFrame(a))
        if "Route-Map" in check:
            st.header("Route-Map")
            b = VisualizationMap(geo=pos_text,key=OSM_key,lim_time=time_day,method_num=method_num).map()
            st.components.v1.html(folium.Figure().add_child(b).render(), height=500)
    else:
        st.sidebar.write("Required items are missing")