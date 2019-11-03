from bokeh.core.properties import Instance
from bokeh.models import ColumnDataSource, Tool


scroll_state_impl = """
import * as p from "core/properties"
import {GestureTool, GestureToolView} from "models/tools/gestures/gesture_tool"

export class ScrollStateToolView extends GestureToolView

  # executed when scrolling
  _scroll: (e) ->
    @model.source.data = {new_state: [True]}
    return null

export class ScrollStateTool extends GestureTool
  default_view: ScrollStateToolView
  type: "ScrollStateTool"

  tool_name: "Scroll state"
  icon: "bk-tool-icon-wheel-zoom"
  event_type: "scroll"
  default_order: 12

  @define { source: [ p.Instance ] }
"""

class ScrollStateTool(Tool):
    __implementation__ = scroll_state_impl
    source = Instance(ColumnDataSource)
