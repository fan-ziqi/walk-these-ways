import sys, select, termios, tty

settings = termios.tcgetattr(sys.stdin)
key = ''
def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key
x_vel_cmd = 0
y_vel_cmd = 0
yaw_vel_cmd = 0
try:
    print("Reading from the keyboard!")
    while key != '\x03':  # CTRL-C
        key = getKey()
        if key == 'w':
            x_vel_cmd += 0.1
        elif key == 's':
            x_vel_cmd -= 0.1
        elif key == 'a':
            y_vel_cmd += 0.1
        elif key == 'd':
            y_vel_cmd -= 0.1
        elif key == 'z':
            yaw_vel_cmd += 0.1
        elif key == 'c':
            yaw_vel_cmd -= 0.1
        else:
            x_vel_cmd = 0
            y_vel_cmd = 0
            yaw_vel_cmd = 0
            if key == '\x03':
                break
        print(str(x_vel_cmd) + " " + str(y_vel_cmd) + " " + str(yaw_vel_cmd) )

except Exception as e:
    print(e)
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
