import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BODYPART_LINES = [('ankle1', 'knee1', 'ro-'), 
                  ('knee1', 'hip1', 'go-'),
                  ('hip1', 'shoulder1', 'yo-'),
                  ('shoulder1', 'elbow1', 'co-'),
                  ('elbow1', 'wrist1', 'mo-'),
                  ('ankle2', 'knee2', 'ro-'), 
                  ('knee2', 'hip2', 'go-'),
                  ('hip2', 'shoulder2', 'yo-'),
                  ('shoulder2', 'elbow2', 'co-'),
                  ('elbow2', 'wrist2', 'mo-'),
                  ('hip1', 'hip2', 'yo-'),
                  ('shoulder1', 'shoulder2', 'co-')]
DEFAULT_BODYPARTS = ['ankle1', 'knee1', 'hip1', 'shoulder1', 'elbow1', 'wrist1', 'ankle2', 'knee2', 'hip2', 'shoulder2', 'elbow2', 'wrist2']
COLORS = ['ro-', 'go-', 'yo-', 'co-', 'mo-', 'bo-']

def continues_line(p1, p2):
    ''' Plot a continues line. '''
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    l = mlines.Line2D([xmin,xmax], [ymin,ymax], linestyle='--')
    ax.add_line(l)
    
def plot_continues_line(df, bodypart1, bodypart2, frame=None):
    if frame == None:
        x1 = df[bodypart1]['x'].mean()
        y1 = df[bodypart1]['y'].mean()
        x2 = df[bodypart2]['x'].mean()
        y2 = df[bodypart2]['y'].mean()
    else:
        x1 = df[bodypart1]['x'][frame]
        y1 = df[bodypart1]['y'][frame]
        x2 = df[bodypart2]['x'][frame]
        y2 = df[bodypart2]['y'][frame]
    continues_line((x1, y1), (x2, y2))

def plot_bodypart_lines(df, frame = None, bodypart_lines = BODYPART_LINES):
    for line in bodypart_lines:
        if frame == None:
            x1 = df[line[0]]['x'].mean()
            y1 = df[line[0]]['y'].mean()
            x2 = df[line[1]]['x'].mean()
            y2 = df[line[1]]['y'].mean()
        else:
            x1 = df[line[0]]['x'][frame]
            y1 = df[line[0]]['y'][frame]
            x2 = df[line[1]]['x'][frame]
            y2 = df[line[1]]['y'][frame]
        plt.plot([x1, x2], [y1, y2], line[2])
    
def plot_bodypoint(df, bodypart, color, alpha = 0.1, frame = None):
    if frame == None:
        x = df[bodypart]['x'].mean()
        y = df[bodypart]['y'].mean()
    else:
        x = df[bodypart]['x'][frame]
        y = df[bodypart]['y'][frame]
    plt.plot(x, y, color, alpha=alpha)

def stick_movie(df):
    ''' Create a stick figure movie for the XY data. '''
    df = -1*df
    fig = plt.figure(figsize=(10,12))
    for i in range(len(df)): 
        plot_bodypart_lines(df, frame=i)
        fig.canvas.draw()
        plt.pause(0.0000000001)
        plt.clf()
    plt.close('all')