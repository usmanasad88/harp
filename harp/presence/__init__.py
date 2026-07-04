"""Presence: is anyone actually in front of the robot?

A cheap, always-on webcam check that drives sleep/wake so the expensive cloud
session only runs when there's a human to talk to. Publishes PresenceChanged;
subscribes to nothing. Deliberately lighter than face-ID (just "someone / no
one", not "who"), so it can run continuously at negligible cost.
"""
