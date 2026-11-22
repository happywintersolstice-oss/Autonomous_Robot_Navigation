"""Free-square selectors that also support precise coordinate entry."""
import streamlit as st


def pick_cells(task, keys, prefix, default):
    """Use only walkable squares in the everyday picker; preserve form edits."""
    for key,value in zip(keys,default,strict=True):
        if key not in st.session_state:
            st.session_state[key] = value
    cells = task.grid.free_cells
    st.write('Choose two open squares using the row and column numbers printed on the map. Walls are excluded from these lists.')
    for column,label,pair in zip(st.columns(2),('Starting square','Destination'),(keys[:2],keys[2:]),strict=True):
        picker = f'{prefix}_{task.grid.map_id}_{label}'
        current = tuple(st.session_state[key] for key in pair)
        if st.session_state.get(picker) not in cells:
            st.session_state[picker] = current if current in cells else cells[0]
        def changed(widget_key=picker, coordinate_keys=pair):
            for key,value in zip(coordinate_keys,st.session_state[widget_key],strict=True):
                st.session_state[key] = value
        column.selectbox(label,cells,key=picker,on_change=changed,
                         format_func=lambda cell:f'Row {cell[0]}, column {cell[1]}')
    with st.expander('Preview starting square and destination'):
        from .interface import show_map
        start = [st.session_state[key] for key in keys[:2]]
        goal = [st.session_state[key] for key in keys[2:]]
        preview = {'scenario':{'start':start,'goal':goal},
                   'initial_observation':[*start,*goal], 'transitions':[]}
        st.caption('Route preview only. No moves have been generated yet.')
        show_map(task.grid,preview,0)


def coordinate_inputs(keys, height, width):
    with st.expander('Enter exact coordinates (optional)'):
        a,b = st.columns(2)
        sr = a.number_input('Start row',0,height-1,key=keys[0])
        sc = a.number_input('Start column',0,width-1,key=keys[1])
        gr = b.number_input('Goal row',0,height-1,key=keys[2])
        gc = b.number_input('Goal column',0,width-1,key=keys[3])
    return sr,sc,gr,gc
