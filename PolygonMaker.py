import folium
from folium.plugins import MarkerCluster, MiniMap, Draw, BeautifyIcon, MeasureControl, MousePosition, Geocoder
from os import system

iranCenter = (33.91605005732569, 53.985586420827794)

Tiles = ['CartoDB positron', 'Stamen Terrain', 'Stamen Toner', 
'OpenStreetMap', 'Stamen Watercolor', 'CartoDB Dark_matter']
iranMap = folium.Map(iranCenter, zoom_start=5, tiles=Tiles[3], control_scale=True)

minimap = MiniMap(toggle_display=True)
iranMap.add_child(minimap)
draw = Draw(export=True)
draw.add_to(iranMap)
MousePosition().add_to(iranMap)
folium.LayerControl().add_to(iranMap)
MeasureControl().add_to(iranMap)
Geocoder(position='bottomright').add_to(iranMap)

iranMap.save('Maps/Iran Map.html')
system('google-chrome "Maps/Iran Map.html"')