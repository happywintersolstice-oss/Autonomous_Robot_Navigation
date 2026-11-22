"""Exercise the same three-route navigation used by a person."""
from robot_navigation.dashboard import NAVIGATION


def benchmark_app(*args,**kwargs):
    """Historical evidence tests explicitly select their original benchmark map."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(*args,**kwargs)
    app.session_state['demo_map'] = 'bottleneck-v1'
    return app


def navigate(app,page):
    record = next(row for row in NAVIGATION if row[0]==page)
    next(button for button in app.button if button.label==record[2]).click().run()
    if record[2]=='Results':
        next(widget for widget in app.radio if widget.label=='Benchmark').set_value(record[3]).run()
        if record[3]=='Fixed layout':
            next(widget for widget in app.selectbox if widget.label=='Results view').set_value(page).run()
    return app
