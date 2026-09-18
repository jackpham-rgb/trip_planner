# Travel Recommendation Platform

## Purpose and aim

## The core idea

Jack has spent months building a detailed Excel travel database covering full length trips (two weeks to a month, region by region) and, more recently, a short trip section for one, two, and three day getaways from Orange County. Both are excellent references, but they are static. Opening the file and deciding what to do next still takes real effort: scrolling through tabs, remembering what has already been done, weighing options by hand.

The new goal is to turn that static reference into a running app or platform: something Jack opens the way he would open a weather app, gets asked a short set of questions, and receives a ranked short list of what to do next. Over time it should also remember what he has already done and whether he wants to repeat it, so the same good suggestion does not have to be rediscovered by hand every time, and a genuinely loved activity is not permanently buried just because it happened once.

## How it should feel to use

1. Ask what kind of trip this is. Multiple choice: a long trip, a day trip, just hanging out nearby, or a specific duration if Jack wants to type one in.
2. Ask a short set of relevance questions about what is actually on his mind right now: mood, energy level, budget, who is coming, indoor or outdoor, adventurous or relaxed, and so on. Multiple choice throughout, not open text, so it stays fast to answer.
3. Return a ranked list of the best next suggestions: specific activities or places, not just categories, pulled from the option bank and scored against the answers above.
4. Let Jack mark something as done, and separately ask or record whether he would want to repeat it. Done and not wanted again should drop out of future suggestions. Done and would repeat should stay in the pool, just weighted differently than something brand new.
5. Let Jack add to, edit, or manage the option bank directly. The option bank should also be able to grow on its own over time, similar to how open source software gets community updates, so the pool of ideas does not stay frozen at whatever was typed in on day one.

## Why this is a Berkeley EECS project, not just an app

Jack wants this built using ideas from his own coursework rather than a generic recommendation library, both because it will teach him something and because it is a stronger portfolio piece. The rough mapping between coursework and the pieces of this system:

Probability, as covered in a course like CS70, is the natural language for saying how confident a suggestion is and for updating that confidence as Jack accepts, rejects, or repeats activities. Every recommendation is really a belief about what Jack will enjoy, and that belief should update the more it observes.

State spaces and Markov style thinking, also touched in CS70, are a natural way to represent a trip or an evening out as a sequence of states and transitions rather than a single flat choice. Where Jack is right now (just got home, tired, hungry, free until Sunday) is a state, and the recommendation is really a transition to a better state.

Google style PageRank is a natural way to rank the option bank itself. Instead of links between web pages, the graph could connect activities that tend to get chosen together, or places that are near each other, or categories that share a mood, and importance flows through that graph the same way it does across the web. A version of this personalized toward Jack's own history, sometimes called a personalized or topic sensitive PageRank, is probably the more useful variant here.

Calculus and differential equations are a natural way to model things that change smoothly over time rather than in sudden jumps: interest in a repeated activity fading after it was just done, then slowly recovering the longer it has been, the same shape as a decay and recovery curve. A course like CS189 adds the machine learning side on top: turning activities and past choices into features and scores rather than hand written rules once there is enough history to learn from.

CS162, or an equivalent systems course, is the reason this is called a platform rather than just a script. A real system needs a clear separation between the content (the option bank, which should be freely updatable) and the state (what Jack has done, rated, and wants to repeat, which must never be silently overwritten by a content update). It also needs to actually run somewhere, sync across devices if there is more than one, and handle updates to the option bank without corrupting history. Any other coursework Jack has in mind for control loops or feedback (for example treating the accept and reject signal as an error term that adjusts future weights) fits naturally alongside this.

## What is intentionally left open

This document is deliberately a statement of purpose, not a finished design. The point of sending it to another agent alongside the current setup document and the Excel workbook is to get a second, independent pass at the architecture: how the questions should actually be structured, what the scoring function should look like in more detail, what the data model for the option bank should be, whether this should be a phone app, a web app, or something else, and how much of the academic mapping above is a genuinely good fit versus a nice sounding idea that does not hold up under implementation.

Please read this together with the current setup document and the workbook, and then push further than what is written here. Propose additional angles, additional coursework or techniques that could apply, alternative architectures, and anything about the intake questions, the scoring model, or the option bank design that has not been considered yet. Treat everything above as a starting draft to be improved on, not a spec to be followed literally.
