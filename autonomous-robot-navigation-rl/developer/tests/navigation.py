"""Exercise the same three-route navigation used by a person."""
from robot_navigation.dashboard import NAVIGATION


def navigate(app,page):
    record = next(row for row in NAVIGATION if row[0]==page)
    next(button for button in app.button if button.label==record[2]).click().run()
    if record[2]=='Results':
        next(widget for widget in app.radio if widget.label=='Benchmark').set_value(record[3]).run()
        if record[3]=='Fixed layout':
            next(widget for widget in app.selectbox if widget.label=='Results view').set_value(page).run()
    return app
