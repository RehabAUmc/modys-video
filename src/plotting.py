import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def newline(p1, p2):
    ''' Plot a line. '''
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
        
def plot_body_lines(ankle_right_x, ankle_right_y, knee_right_x, knee_right_y, hip_right_x, hip_right_y, 
            wrist_right_x, wrist_right_y, elbow_right_x, elbow_right_y, shoulder_right_x, shoulder_right_y, 
            ankle_left_x, ankle_left_y, knee_left_x, knee_left_y, hip_left_x, hip_left_y, 
            wrist_left_x, wrist_left_y, elbow_left_x, elbow_left_y, shoulder_left_x, shoulder_left_y):
    ''' Plot all body limbs. '''
    plt.plot([ankle_right_x, knee_right_x], [ankle_right_y, knee_right_y], 'ro-')
    plt.plot([knee_right_x, hip_right_x], [knee_right_y, hip_right_y], 'go-')
    plt.plot([hip_right_x, shoulder_right_x], [hip_right_y, shoulder_right_y], 'yo-')
    plt.plot([shoulder_right_x, elbow_right_x], [shoulder_right_y, elbow_right_y], 'co-')
    plt.plot([elbow_right_x, wrist_right_x], [elbow_right_y, wrist_right_y], 'mo-')

    plt.plot([ankle_left_x, knee_left_x], [ankle_left_y, knee_left_y], 'ro-')
    plt.plot([knee_left_x, hip_left_x], [knee_left_y, hip_left_y], 'go-')
    plt.plot([hip_left_x, shoulder_left_x], [hip_left_y, shoulder_left_y], 'yo-')
    plt.plot([shoulder_left_x, elbow_left_x], [shoulder_left_y, elbow_left_y], 'co-')
    plt.plot([elbow_left_x, wrist_left_x], [elbow_left_y, wrist_left_y], 'mo-')

    plt.plot([hip_right_x, hip_left_x], [hip_right_y, hip_left_y], 'bo-')
    plt.plot([shoulder_right_x, shoulder_left_x], [shoulder_right_y, shoulder_left_y], 'bo-')
    
    newline([shoulder_right_x, shoulder_right_y],[hip_right_x, hip_right_y])
    newline([shoulder_left_x, shoulder_left_y],[hip_left_x, hip_left_y])

def stick_movie(df):
    ''' Create a stick figure movie for the XY data. '''
    df = -1*df
    fig = plt.figure(figsize=(10,12))
    for i in range(len(df)):
        plt.xlim(-15,-2)   
        plt.ylim(-15,-0) 
        plot_body_lines(df.iloc[i]['ankle1'], df.iloc[i]['ankle1.1'], df.iloc[i]['knee1'], df.iloc[i]['knee1.1'], df.iloc[i]['hip1'], df.iloc[i]['hip1.1'],
                    df.iloc[i]['wrist1'], df.iloc[i]['wrist1.1'], df.iloc[i]['elbow1'], df.iloc[i]['elbow1.1'], df.iloc[i]['shoulder1'], df.iloc[i]['shoulder1.1'], 
                    df.iloc[i]['ankle2'], df.iloc[i]['ankle2.1'], df.iloc[i]['knee2'], df.iloc[i]['knee2.1'], df.iloc[i]['hip2'], df.iloc[i]['hip2.1'],
                    df.iloc[i]['wrist2'], df.iloc[i]['wrist2.1'], df.iloc[i]['elbow2'], df.iloc[i]['elbow2.1'], df.iloc[i]['shoulder2'], df.iloc[i]['shoulder2.1'])
        fig.canvas.draw()
        plt.pause(0.0000000001)
        plt.clf()
    plt.close('all')