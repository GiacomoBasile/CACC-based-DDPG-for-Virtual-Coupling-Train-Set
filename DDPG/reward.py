from math import cos, sin
from config import PI

def reward(speedX, angle, trackPos):

	if trackPos > 1 or trackPos < -1:
		return -200
	else:
		return speedX*cos(angle) - speedX*abs(sin(angle)) - speedX*abs(trackPos)

def reward_notrackPos(speedX, angle, trackPos):

	if trackPos > 1 or trackPos < -1:
		return -200
	else:
		return speedX*cos(angle) - speedX*abs(sin(angle))

def reward_angle(speedX, angle, trackPos):

	if trackPos > 1 or trackPos < -1:
		return -200
	else:
		return speedX*cos(angle) - speedX*abs(sin(angle)) - speedX*abs(angle/PI)