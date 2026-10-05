# Instructions for running code
To run the main code with a simeple ROA sweep
uv run assignment_1.py 

To run poincare simulation giving a one dimensional return map
uv run assignment_1_poincare.py

To run a sweep of the ROA depending on the angle of the slope and the number so spokes
uv run assignment_1_sweep.py

# Sanity Checks
Does the wheel stay at rest at a slope of 0 degrees?
Expected behavior is that after falling to have two legs in contact with the slope the wheel will not have any significant kinetic energy and will rapidly shift between which legs persepective it is working from.

Does the wheel stay at rest at a starting angle of 0 degrees?
Expected behavior is that the wheel will not move at all and will have constant potential energy and no kinetic energy.

Does the wheel stay at rest when it starts at contact angle?
Expected behavior is the wheel will not have any significant kinetic energy and will rapidly shift between which legs persepective it is working from therefore shifting the potential energy.

Does the wheel initially accelerate up the slope when starting at a negative angle?
Expected behavior is the wheel will have negative angular velocity for a period before one of the legs contacts the slope.

Can the wheel reach a steady state going down the slope?
Expected behavior is that when decending down a sufficiently steep slope with sucifficient initial energy the wheel will reach a steady state cycle going down the slope.

# ROA Findings
The region of attraction for the limit cycle covers most of the space outside of the x axis as the initial angular velocity prevents the stability of the system. Everytime the system is not in a stationary stable position in converges to the same limit cycle in the state space.

# Return Map Findings
The return map operated as expected but did reveal some issues with how the simulation was being conducted that I had to solve but the refined simulation allowed for larger timesteps increasing my simulation speed

# Spokes and Slope Sweep
As the number of spokes increased the amount of times the wheel converged to rest decreased as less energy was required to get the wheel to the next leg. As the slopes angle increased less simulations converged to rest because there was more potential energy from the slope although this did reveal some inaccuracies in my simulation since extreme slopes should cause slipping issues.