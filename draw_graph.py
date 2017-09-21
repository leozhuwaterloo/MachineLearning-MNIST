import plotly.plotly as py
import plotly.graph_objs as go
import plotly


def draw_graph(xs, ys, title="Success Rate"):
    plotly.tools.set_credentials_file(username='yuner25699', api_key='noNGMtSisHSfKLayAvWf')

    graph_data = [
        go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            name='lines',
        ),
        go.Scatter(
            x=xs,
            y=ys,
            visible='legendonly',
            mode='markers',
            name='markers',
        ),
        go.Scatter(
            x=xs,
            y=ys,
            visible='legendonly',
            mode='lines+markers',
            name='lines+markers',
        ),
    ]

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis=dict(
            title='Number of Batches',
        ),
        yaxis=dict(
            title='Success Rate',
        ),
        showlegend=True
    )

    fig = go.Figure(data=graph_data, layout=layout)
    py.plot(fig)


def draw_graph_3d(xses, yses, zses, names, title="Success Rate"):
    plotly.tools.set_credentials_file(username='yuner25699', api_key='noNGMtSisHSfKLayAvWf')
    n = len(xses)
    graph_data = []
    for i in range(n):
        graph_data.extend([
            go.Scatter3d(
                x=xses[i],
                y=yses[i],
                z=zses[i],
                mode='lines',
                name='lines-' + names[i],
            ),
            go.Scatter3d(
                x=xses[i],
                y=yses[i],
                z=zses[i],
                visible='legendonly',
                mode='markers',
                name='markers-' + names[i],
            ),
            go.Scatter3d(
                x=xses[i],
                y=yses[i],
                z=zses[i],
                visible='legendonly',
                mode='lines+markers',
                name='lines+markers-' + names[i],
            ),
        ])

    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='Number of Neutrons in Hidden Layer',
            ),
            yaxis=dict(
                title='Number of Batches',
            ),
            zaxis=dict(
                title='Success Rate',
            ),
        ),
        title=title,
        hovermode='closest',
        showlegend=True
    )

    fig = go.Figure(data=graph_data, layout=layout)
    py.plot(fig)
