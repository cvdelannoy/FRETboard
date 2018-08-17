%%opts Curve [tools=['box_select']]

rects_opts = dict(color=hv.Palette('Category10'), color_index='level', line_width=0.000001)
curve_opts = dict(height=250, width=800, xaxis=None, tools=['box_select'], color='black', line_width=1)

def rectangle(x=0, y=0, width=1, height=1):
    return np.array([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])

# Train hmm
with warnings.catch_warnings():  # catches deprecation warning sklearn: log_multivariate_normal_density
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    hmm.train()


# Declare time series
example = hmm.data.loc[np.invert(hmm.data.is_labeled)].sample(1)
example.labels = example.prediction
i_fret = example.i_fret.values[0]
ts = hv.Curve((np.arange(i_fret.size), i_fret)).options(**curve_opts)
tp = hv.Points((np.arange(i_fret.size), i_fret)).options(**curve_opts)

# Declare stream
selection = streams.Selection1D(source=tp)

# Function that reacts to selection
def display_selection(sel_state, index):
    if index:
        example.prediction.values[0][index] = sel_state
    pred = example.prediction.values[0] / nb_states
    time = np.arange(i_fret.size)
    pred = np.column_stack((time, pred))
    pred_rects = hv.Polygons([{('x', 'y'): rectangle(x), 'level': z} for
                                  x, z in pred], vdims='level').options(**rects_opts)
    return pred_rects

# Declare options for a new state selection
nb_events_choices = list(range(hmm.nb_states))

# Plot

dm = hv.DynamicMap(display_selection, kdims=['sel_state'], streams=[selection]).redim.values(sel_state=nb_events_choices)
dm * ts * tp

