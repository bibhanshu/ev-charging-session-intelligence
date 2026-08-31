##Problem statement :
"""
Stage 2 -- Station Availability / Congestion Classification -- CONCEPT NOTES
================================================================================

Training the model (what we're about to do): we look at all your historical
sessions (Sept 2018-Mar 2019) and, for every station-hour combination that
already happened, calculate: was it occupied or not, based on how many
sessions' connect-disconnect windows overlapped that hour. This gives us
historical patterns -- e.g., "Station AG-3F32 is occupied 70% of 9am hours
on weekdays, historically." The model learns these patterns from station,
hour, day-of-week, month.

Using the trained model to predict (later, in production/the app): once
trained, the model does NOT need live data of who's currently connected.
Instead, it predicts based on the pattern it learned -- you'd ask it
"Station AG-3F32, Tuesday, 9am" and it answers "historically, this is
usually busy" -- a probabilistic forecast based on learned patterns, not
a live status check.

The simplest way to think about it: this model isn't answering "is someone
plugged in right now" (that would need live sensor data, which we don't
have and isn't the point). It's answering "based on historical patterns,
how likely is this station to be busy at this day/hour?" -- like a weather
forecast. A weather model doesn't need live real-time data to tell you
"Tuesdays in July are usually hot in this city" -- it learned that from
historical patterns. Same idea here.

We are looking to answer:
"How busy will this station be, based on historical patterns of everyone
who's used it" -> this is genuine Stage 2 congestion, needs station+time
features only (stationID, clusterID, hour, day-of-week, month) -- NOT the
individual driver's own milesRequested/minutesAvailable, since those only
explain that one driver's own session, not the collective pattern of
everyone who uses that station.
"""