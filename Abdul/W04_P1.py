import folium

t_list = ["Stamen Terrain", "Stamen Toner", "Mapbox Bright"]
map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = t_list[0],
                        zoom_start = 12)
map_hooray



folium.Marker([51.5079, 0.0877],
              popup='London Bridge',
              ).add_to(map_hooray)
map_hooray



folium.Marker([51.5079, 0.0877],
              popup='London Bridge',
              icon=folium.Icon(color='green')
              ).add_to(map_hooray)

map_hooray


## see more here: https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-data