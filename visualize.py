import json
from typing import List

import colorcet

import bokeh.document
import bokeh.embed.bundle
import bokeh.embed
import bokeh.io
import bokeh.models
import bokeh.plotting
import bokeh.transform
import bokeh.resources
import numpy as np


class ExtensibleResources(bokeh.resources.Resources):

    def __init__(self, additional_js_files):
        self._additional_js_files = additional_js_files
        super().__init__()

    @property
    def js_files(self) -> List[str]:
        return super().js_files + self._additional_js_files


def visualize():
    with open("out.json") as file:
        json_import = json.load(file)

    edo = json_import["edo"]
    scale_size = json_import["scale_size"]
    min_step = json_import["min_step"]
    max_step = json_import["max_step"]
    points = np.array(json_import["points"])
    scales = json_import["scales"]
    consonances = json_import["consonances"]
    normalized_consonances = json_import["normalized_consonances"]

    title = f"UMAP of {len(scales)} most consonant {scale_size}-tone {edo}EDO scale classes, step range {min_step}-{max_step}"

    strings = ["{" + ", ".join([str(x) for x in scale]) + "}" for scale in scales]

    source = bokeh.models.ColumnDataSource(data={
        "x": points[:, 0],
        "y": points[:, 1],
        "scale": scales,
        "consonance": consonances,
        "normalized_consonance": json_import["normalized_consonances"],
        "string": strings,
    })

    hover_tool = bokeh.models.HoverTool(
        tooltips=[
            ("scale", "@string"),
            ("consonance", "@consonance"),
        ],
    )

    custom_js = bokeh.models.CustomJS(
        args={"source": source, "edo": edo},
        code="""
            const scale = source.data.scale[source.selected.indices[0]];
            playScale(scale, edo);
        """
    )
    tap_tool = bokeh.models.TapTool(callback=custom_js)

    default_tools = ["pan", "wheel_zoom", "box_zoom", "save", "reset", "help"]
    tools = [hover_tool, tap_tool] + default_tools
    plot = bokeh.plotting.figure(
        title=title,
        width=800,
        height=800,
        tools=tools,
    )
    plot.grid.visible = False
    plot.axis.visible = False

    normalize = True
    if normalize:
        color_field_name = "normalized_consonance"
    else:
        color_field_name = "consonance"

    plot.circle(
        source=source,
        color=bokeh.transform.linear_cmap(
            field_name=color_field_name,
            palette=colorcet.CET_L18[::-1],
            low=min(source.data[color_field_name]),
            high=max(source.data[color_field_name]),
        ),
        size=3.0,
        alpha=0.5,
    )

    resources = ExtensibleResources(
        additional_js_files=["Tone.js", "app.js"]
    )

    document = bokeh.document.Document()
    document.add_root(plot)
    document.theme = "dark_minimal"
    document.title = title

    html = bokeh.embed.file_html(document, resources)
    with open("out.html", "w") as file:
        file.write(html)


if __name__ == "__main__":
    visualize()
