"""Warehouse floor-plan annotations; geometry and policy decisions stay separate."""
from functools import lru_cache

from matplotlib.patches import Rectangle

from .core import ROOT,read_json
from .warehouse import MAP_ID


@lru_cache(maxsize=4)
def layout_for_grid(grid):
    if grid.map_id!=MAP_ID:
        return None
    layout = read_json(ROOT/'data/warehouse'/MAP_ID/'map.json')
    if tuple(layout['rows'])!=grid.rows:
        raise ValueError('Warehouse annotations do not match this map geometry')
    return layout


def station_name(grid,cell):
    layout = layout_for_grid(grid)
    if layout is not None:
        for item in layout['stations']:
            if tuple(item['cell'])==tuple(cell):
                return item['name']
    return f'Floor square {tuple(cell)}'


def decorate(ax,grid,episode):
    layout = layout_for_grid(grid)
    if layout is None:
        return
    for zone in layout['zones']:
        r1,c1,r2,c2 = zone['bounds']
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                if grid.is_free((r,c)):
                    ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=zone['color'],edgecolor='none',zorder=.2))
        ax.text((c1+c2)/2,r2+.15,zone['name'],ha='center',va='center',fontsize=8,
                weight='bold',color='#b5d4e6',zorder=3)
    for item in layout['fixtures']:
        r1,c1,r2,c2 = item['bounds']
        ax.add_patch(Rectangle((c1-.46,r1-.46),c2-c1+.92,r2-r1+.92,
                               facecolor='#243e50',edgecolor='#5e8296',linewidth=1.4,zorder=1))
        if item['kind']=='storage rack':
            for row in range(r1,r2+1):
                ax.plot([c1-.4,c2+.4],[row,row],color='#4e6e82',linewidth=.7,zorder=1.1)
        label = item['name'] if item['kind']=='storage rack' else 'BENCH' if item['kind']=='work bench' else 'PALLETS'
        ax.text((c1+c2)/2,(r1+r2)/2,label,ha='center',va='center',
                rotation=90 if item['kind']=='storage rack' else 0,fontsize=8,
                color='#dceaf3',weight='bold',zorder=3.2,
                bbox={'facecolor':'#243e50','edgecolor':'none','pad':1})
    occupied = {tuple(episode['scenario']['start']),tuple(episode['scenario']['goal'])}
    for item in layout['stations']:
        r,c = item['cell']
        ax.add_patch(Rectangle((c-.3,r-.3),.6,.6,facecolor='none',edgecolor='#e7c474',
                               linewidth=1.2,linestyle=':',zorder=1.5))
        if tuple(item['cell']) not in occupied:
            code = {'receiving':'IN','packing':'PK','dispatch':'OUT','charging':'CHG',
                    'rack-a':'A','rack-b':'B','rack-c':'C'}[item['id']]
            ax.text(c,r,code,ha='center',va='center',fontsize=7,color='#e7c474',zorder=1.6)


def scene_annotations(grid):
    """Return floor tints, fixture names and station labels for the 3D renderer."""
    layout = layout_for_grid(grid)
    if layout is None:
        return {},{},[]
    colors,fixtures,labels = {},{},[]
    height,_ = grid.shape
    for zone in layout['zones']:
        r1,c1,r2,c2 = zone['bounds']
        color = [int(zone['color'][i:i+2],16) for i in (1,3,5)]
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                if grid.is_free((r,c)):
                    colors[(r,c)] = color
        labels.append({'position':[(c1+c2)/2+.5,height-r2-.7,.12],
                       'text':zone['name'],'color':[181,212,230],'size':12})
    for item in layout['fixtures']:
        r1,c1,r2,c2 = item['bounds']
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                fixtures[(r,c)] = item['name']
        labels.append({'position':[(c1+c2)/2+.5,height-(r1+r2)/2-.5,.85],
                       'text':item['name'],'color':[220,234,243],'size':12})
    for item in layout['stations']:
        r,c = item['cell']
        code = {'receiving':'IN','packing':'PK','dispatch':'OUT','charging':'CHG',
                'rack-a':'A','rack-b':'B','rack-c':'C'}[item['id']]
        labels.append({'position':[c+.5,height-r-.5,.18],'text':code,
                       'color':[231,196,116],'size':10})
    return colors,fixtures,labels
