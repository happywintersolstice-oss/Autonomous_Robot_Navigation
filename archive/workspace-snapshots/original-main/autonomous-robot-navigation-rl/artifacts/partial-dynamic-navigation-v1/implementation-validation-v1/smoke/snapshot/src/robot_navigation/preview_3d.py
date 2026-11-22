"""Temporary 3D presentation of existing grid maps and displayed replay prefixes.

Coordinates remain grid cells. Raised walls are a visual extrusion, not a new
navigation environment. This module never loads policies or changes episodes.
"""
from collections import Counter
from math import log2

import pydeck as pdk

from .episodes import frame


def scene_data(grid, episode, index, trail=True, visits=False, bfs=False):
    """Project only the visible frame; future recorded moves are never exposed."""
    height, width = grid.shape
    state = frame(episode,index)
    from .warehouse_scene import scene_annotations
    floor_colors,fixture_names,warehouse_labels = scene_annotations(grid)

    def position(cell,z=.08):
        row,column = cell
        return [column+.5,height-row-.5,z]

    floor,walls,labels = [],[],[]
    for row,values in enumerate(grid.rows):
        for column,value in enumerate(values):
            y = height-row-1
            cell = {'polygon':[[column,y],[column+1,y],[column+1,y+1],[column,y+1]],
                    'color':floor_colors.get((row,column),[11,29,41]),
                    'description':f"{fixture_names.get((row,column),'Wall' if value=='#' else 'Open square')}: row {row}, column {column}"}
            floor.append(cell)
            if value=='#':
                walls.append(cell)
    start,goal = episode['scenario']['start'],episode['scenario']['goal']
    markers = [{'position':position(cell,.12),'color':color,'radius':radius,'description':description}
               for cell,color,radius,description in (
                   (start,[139,166,184],.20,'S: starting square'),
                   (goal,[0,223,181],.25,'G: destination'),
                   (state['position'],[0,189,242],.19,'R: current robot position'))]
    # Combine labels when the robot occupies a start/goal square.
    grouped = {}
    for cell,label in ((start,'S'),(goal,'G'),(state['position'],'R')):
        grouped.setdefault(tuple(cell),[]).append(label)
    for cell,names in grouped.items():
        labels.append({'position':position(cell,.9),'text':' / '.join(names),
                       'color':[220,242,255],'size':15})
    for row in range(height):
        labels.append({'position':[-.55,height-row-.5,.02],'text':str(row),
                       'color':[136,169,191],'size':11})
    for column in range(width):
        labels.append({'position':[column+.5,-.55,.02],'text':str(column),
                       'color':[136,169,191],'size':11})
    labels.extend([
        {'position':[-1.3,height/2,.02],'text':'Row','color':[136,169,191],'size':12},
        {'position':[width/2,-1.2,.02],'text':'Column','color':[136,169,191],'size':12}])
    labels.extend(warehouse_labels)
    if visits:
        for cell,count in Counter(map(tuple,state['trail'])).items():
            x,y,z = position(cell,.15)
            labels.append({'position':[x+.25,y-.25,z],'text':str(count),
                           'color':[141,229,221],'size':11})
    paths = []
    if trail and len(state['trail'])>1:
        paths.append({'path':[position(cell,.04) for cell in state['trail']],
                      'color':[134,221,255],'description':'Route travelled so far'})
    if bfs:
        paths.append({'path':[position(cell,.06) for cell in grid.path(start,goal)],
                      'color':[186,157,255],'description':'Known-map shortest-path reference'})
    collision = []
    if state['status']=='collision':
        attempted = position(state['attempted_position'],.2)
        labels.append({'position':attempted,'text':'X','color':[255,106,106],'size':24})
        collision.append({'source':position(state['position'],.2),'target':attempted,
                          'description':'Recorded collision attempt'})
    return {'floor':floor,'walls':walls,'markers':markers,'labels':labels,
            'paths':paths,'collision':collision,'target':[width/2,height/2,0],
            'zoom':log2(240/max(height,width)),'state':state}


def make_deck(grid,episode,index,trail=True,visits=False,bfs=False):
    data = scene_data(grid,episode,index,trail,visits,bfs)
    # Streamlit's bundled deck.gl expects the named coordinate system. Keep it
    # literal so pydeck does not serialize it as an accessor expression.
    coordinates = pdk.types.String('cartesian')
    pixels = pdk.types.String('pixels')
    common = {'coordinate_system':coordinates,'pickable':True,'auto_highlight':True}
    layers = [
        pdk.Layer('PolygonLayer',id='floor',data=data['floor'],get_polygon='polygon',
                  get_fill_color='color',get_line_color=[53,83,104],
                  stroked=True,get_line_width=1,line_width_units=pixels,**common),
        pdk.Layer('PolygonLayer',id='walls',data=data['walls'],get_polygon='polygon',
                  extruded=True,get_elevation=.65,get_fill_color=[47,89,112],
                  get_line_color=[96,171,195],wireframe=True,**common),
        pdk.Layer('PathLayer',id='trail',data=data['paths'],get_path='path',
                  get_color='color',get_width=3,width_units=pixels,
                  cap_rounded=True,joint_rounded=True,**common),
        pdk.Layer('ScatterplotLayer',id='markers',data=data['markers'],get_position='position',
                  get_fill_color='color',get_radius='radius',radius_units=pdk.types.String('common'),
                  stroked=True,get_line_color=[220,242,255],get_line_width=1,
                  line_width_units=pixels,**common),
        pdk.Layer('LineLayer',id='collision',data=data['collision'],get_source_position='source',
                  get_target_position='target',get_color=[255,106,106],get_width=3,**common),
        pdk.Layer('TextLayer',id='labels',data=data['labels'],get_position='position',
                  get_text='text',get_color='color',get_size='size',billboard=True,
                  font_family=pdk.types.String('Arial'),font_weight=pdk.types.String('bold'),coordinate_system=coordinates),
    ]
    return pdk.Deck(layers=layers,map_provider=None,map_style=None,
                    views=[pdk.View('OrbitView',controller=True,orbit_axis='Z')],
                    initial_view_state=pdk.ViewState(target=data['target'],rotation_x=45,
                        rotation_orbit=30,zoom=data['zoom'],min_zoom=1,max_zoom=7),
                    parameters={'clearColor':[6/255,18/255,28/255,1]},
                    tooltip={'text':'{description}'})
