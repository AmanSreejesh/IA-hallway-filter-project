# IA Hallway Filter Project
Team: Viraj Das, Aman Sreejesh, Ishan Prabhu, Aryan Mali

## How this repo works
- **Issues** hold our user stories. Use the "User story" template.
- **constraints.md** holds the facts every story must respect.
- **test-cases.md** holds tests that trace back to stories by issue number.
- **prototype/** holds snapshots of our code when it passes a story's tests.

## Where code goes
- **viraj/, aman/, ishan/, aryan/** each hold one team member's filter(s). Work only in your own folder.
- **common/main.py** is the main system. It combines everyone's filters and implements the shared features (opt-in zone, fade-in, timer, multi-user, engagement).
- Run everything with `python common/main.py`, or only your filter with `python common/main.py emoji`. See the top of `common/main.py` for how to plug in a filter.

## Labels to create
- user: on purpose, user: passerby, user: maintainer, user: excluded
- priority: high, priority: medium, priority: low
