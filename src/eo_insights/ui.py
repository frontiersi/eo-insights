"""User interfaces for Jupyter Notebooks"""

from types import SimpleNamespace
from typing import Optional
from ipyleaflet import DrawControl, FullScreenControl, Map, WidgetControl
from ipywidgets.widgets import HTML, Button, Layout
from IPython.display import display
from jupyter_ui_poll import run_ui_poll_loop
from odc.geo.geom import Geometry


def make_map_region_selector(
    m: Optional[Map] = None, height: str = "500px", **kwargs
) -> tuple[Map, SimpleNamespace]:
    """
    Method to construct a ipyleaflet.Map with widgets for area selection

    Parameters
    ----------
    m : ipyleaflet.Map, optional
        Existing map, if available, by default None
    height : str, optional
        height of the map, by default "600px"

    Returns
    -------
    tuple[Map, SimpleNamespace]
        The map to display, and the state of the widgets, managed by a SimpleNamespace

        selection: Geometry
            The Geometry of the selected area
        bounds: dict[tuple]
            Coordinates of the selected area in the form {lat: (lat_north, lat_south), lon: (lon_west, lon_east)}
        done: bool
            Status of the widget
    """

    # Set up the state for tracking area selection and status
    state = SimpleNamespace(selection=None, bounds=None, done=False)

    button_done = Button(description="done", layout=Layout(width="5em"))
    button_done.style.button_color = "green"
    button_done.disabled = True

    # Create layout for displaying latitude and longitude bounds
    html_info = HTML(layout=Layout(flex="1 0 20em", width="16em", height="6em"))

    def update_info(text: str):
        """
        Display information in the form of a string using a grey box

        Parameters
        ----------
        text : str
            Information to display
        """
        html_info.value = f'<pre style="color:grey">{text}</pre>'

    def render_bounds(bounds: list[tuple[float, float]]):
        """
        Displays bounds as text

        Parameters
        ----------
        bounds : list[tuple[float, float], tuple[float, float]]
            List of South West and North East location tuples. e.g. [(S, W), (N, E)].
        """
        (lat_south, lon_west), (lat_north, lon_east) = bounds
        text = f"lat: [{lat_south:.{4}f}, {lat_north:.{4}f}]\nlon: [{lon_west:.{4}f}, {lon_east:.{4}f}]"
        update_info(text)

    # Create map if not exists. If map exists, render bounds.
    if m is None:
        m = Map(**kwargs) if len(kwargs) else Map(zoom=2)
        m.scroll_wheel_zoom = True
        m.layout.height = height
    else:
        render_bounds(m.bounds)

    # Display button_done and html_info on the map
    widgets = [
        WidgetControl(widget=button_done, position="topright"),
        WidgetControl(widget=html_info, position="bottomleft"),
    ]
    for w in widgets:
        m.add(w)

    # Add the draw control
    draw = DrawControl()

    # Set as empty to remove from draw control
    draw.circle = {}
    draw.polyline = {}
    draw.circlemarker = {}

    # Set options for rectangles and polygons
    shape_opts = {"fillColor": "#fca45d", "color": "#000000", "fillOpacity": 0.1}

    draw.rectangle = {"shapeOptions": shape_opts, "metric": ["km", "m"]}

    poly_opts = {"shapeOptions": {**shape_opts}}
    poly_opts["shapeOptions"]["original"] = {**shape_opts}
    poly_opts["shapeOptions"]["editing"] = {**shape_opts}

    draw.polygon = poly_opts

    draw.edit = True
    draw.remove = True
    m.add(draw)
    m.add(FullScreenControl())

    def on_done(button):
        state.done = True
        button_done.disabled = True
        m.remove(draw)
        for w in widgets:
            m.remove(w)

    def bounds_handler(event):
        bounds = event["new"]
        render_bounds(bounds)
        (lat_south, lon_west), (lat_north, lon_east) = bounds
        state.bounds = {"lat": (lat_south, lat_north), "lon": (lon_west, lon_east)}

    def on_draw(event):
        drawn_shape = event["new"]
        action = event["name"]
        if action == "last_draw":
            state.selection = drawn_shape["geometry"]
        elif action == "last_action" and drawn_shape == "deleted":
            state.selection = None

        button_done.disabled = state.selection is None

    draw.observe(on_draw)
    m.observe(bounds_handler, ("bounds",))
    button_done.on_click(on_done)

    return m, state


def select_on_a_map(m: Optional[Map] = None, **kwargs):
    """
    Display a map and block execution until user selects a region of interest.

    Parameters
    ----------
    m : ipyleaflet.Map, optional
        Existing map, if available, by default None
    **kwargs
        Any parameter ipyleaflet.Map(..) accepts. Examples:
        height: str
            height of the map, for example "500px", "10el"
        zoom: int
            Zoom level
        center: tuple(float, float)
            Center of the map, as a tuple of (latitude, longitude)

    Returns
    -------
    odc.geo.geom.Geometry
        Geometry that was last drawn on the map at the time of pressing the "done" button
    """

    m_2, state = make_map_region_selector(m=m, **kwargs)
    if m is None:
        display(m_2)

    def ui_poll(f, sleep=0.02, n=1):
        return run_ui_poll_loop(f, sleep, n=n)

    def extract_geometry(state) -> Geometry:
        geom = Geometry(state.selection, crs="EPSG:4326")
        return geom

    return ui_poll(lambda: extract_geometry(state) if state.done else None)
