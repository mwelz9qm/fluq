# Summer 2026 Kickoff — Explainable AI/ML (XAI) for pyMAISE

Welcome aboard, summer crew! Not-so-quick doc to get us all on the same page before we get rolling.

## What we're doing

As you've gathered, we're technically sub-contracting to Majdi Radaideh's group at the University of Michigan on the explainable AI/ML side of the pyMAISE project. You all know the three project areas from spring (or, Kashiva, you're about to know them):

1. MC Dropout for uncertainty quantification
2. Deep Ensembles for uncertainty quantification
3. Variance reduction and bias mitigation via ensembles

After meeting with Majdi, we're focusing on **Projects 2 and 3** this summer and postponing MC Dropout for now. The goal isn't just to extend the spring notebooks — Majdi wants us to build **well-documented, generalizable Python classes** that any pyMAISE user could pick up and apply to their own problem (eventually). Think clean APIs, real docstrings, good visualizations baked in, and the kind of code we'd all be proud to put on GitHub. We'll write these in TensorFlow/Keras for now. As you heard, there's a separate effort underway to move pyMAISE's backend from TF to PyTorch — once our classes are solid, we'll port them, but that's a later problem (that Majdi seems to think is small!).

We'll do **bi-weekly check-ins with Majdi over the summer**. I'll get those on the calendar.

## Working remotely

Most of us are remote this summer, and I've got FLC's MS Teams set up so we have a real workspace. A few honest words about the format:

Synchronous time is going to be limited. Zoom/Teams meetings can be tough to make productive, and honestly, I can't lecture you the way I did sometimes in Junior Design. **This summer is going to lean asynch**, and our approach to work has to match that. Specifically:

- **Teams will be our hub.** I'll set up channels for each team plus a `#general` and `#tech help`. Default to writing things down in the channel, not DMing (which also has its place).
- **Code can be conversation.** Pull requests with real review comments are how we'll do most of the technical back-and-forth. If you've written code and nobody's seen it for three days, something's wrong with out approach.
- **Short, written status updates** Wednesday afternoons each week. One paragraph per person, posted in your team channel: what you did, what you're stuck on (blockers), what's next.
- **Sync meetings when async isn't enough.** I'll start by holding one all-hands Teams meeting per week for announcements and to discuss blockers. Team-internal syncs are up to each team to schedule.

If you're stuck for more than a couple hours, post in the channel. Don't sit on it!

## Teams

**Team 2 — Deep Ensembles**
- Tyler — architect & reviewer
- Grant — day-to-day lead
- Kashiva — onboarding, then full contributor

Tyler, you're theoretically the technical lead on this, but I want to be straight with the team: we know your availability is going to be spotty. So your role looks more like a senior engineer at a company who's part-time on the project — you own the class design and get first review on significant PRs. Grant drives day-to-day and breaks ties when you're offline. This plays to your strengths and doesn't ask you to be in a Teams call all the time.

**Team 3 — Bias & Variance Reduction**
- Dayton — lead (half-time on the project)
- Z — day-to-day driver

Dayton, similar deal but flipped. You've already worked this project in spring and you have a math background it really needs. You lead, but you're half-time, so Z, you're carrying more load. 

The work this summer is going to depend on y'all grinding through things even when nobody's looking over your shoulder. Remote work makes that harder, not easier. But, I trust you can do it!

## Kashiva's onboarding (first ~2 weeks)

Kashiva, you weren't in Junior Design, so we need to get you caught up. Here's the plan:

1. **Junior Design notebooks intro-ing Machine Learning** — I'll share the full set. Work through them in order. They were designed to be lectured alongside, so you will hit spots where things aren't obvious. That's expected, not a sign you're behind!
2. **Pair with Grant** — 30-60 minutes/day, scheduled however works for you two. Grant walks you through one notebook or concept. This is good for him too — explaining things is how you find out what you don't actually understand.
3. **External supplements** — 3Blue1Brown's neural network video series https://www.3blue1brown.com/?topic=neural-networks and Andrew Karpathy's "Neural Networks: Zero to Hero" if you want to go deeper. https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
4. **Capstone for onboarding** — two weeks from Friday, produce your own version of a deep ensemble notebook.

Plan to be at ~50% contribution to Team 2 by week 3 and fully integrated by week 5.

## Week 1 work — everyone

Before anyone touches the actual ML, we have some setup to do. Most of you don't have extensive experience with git or written code anyone else has had to read. We need to remedy that!

1. **GitHub setup.** I'll create the org and a single monorepo (working name: `fluq` — FLC uncertainty quantification — open to better name suggestions!). Get your auth set up (SSH keys or HTTPS with a personal access token — either is fine, SSH is nicer long-term but HTTPS is faster to get going), clone the repo, and push a "hello world" commit to your own branch. The repo will be structured like this:
   ```
   fluq/
     deep_ensembles/       # Team 2's package
     bias_variance/        # Team 3's package
     common/               # shared utilities — data loaders, plotting, metrics
     notebooks/            # example notebooks for each method
     tests/
     README.md
   ```
   Both teams work in the same repo. Shared code (dataset loaders, plots, metrics) lives in `common/` so we don't duplicate work. If you need something that belongs in `common/`, talk to the other team first so we don't end up with two slightly-different versions of the same function.
2. **Git workflow.** I'll post a short written guide in `# tech help`. The non-negotiables: branch off main, open a PR, get a review, then merge. Don't commit to main. Don't force-push to a branch someone else is working on. Since we're all sharing one repo, name your branches descriptively — `team2/ensemble-class-skeleton`, `team3/sampling-bias-study`, `common/chf-loader` — so it's obvious who's working on what.
3. **Environment.** TensorFlow + Keras + Jupyter, matching versions across the team. We'll also need pyMAISE installed for loading datasets — heads up that the install is rough on Mac, I'll eventually pin a message in `#general` for the workaround.
4. **The "what is this project?" doc.** Each of you writes a one-page (max) doc explaining what you think your project is supposed to do and what success looks like. Post in your team channel by next Wednesday. This draws out misunderstandings early, before we've written code based on them.

## Week 1 work (finish by 5/20) — by team

### First, a word on what we may mean by "API"

Both teams' week 1 deliverable is something called an "API skeleton," so let me say what is meant by that, because the term gets thrown around in different ways.

"API" stands for **Application Programming Interface**. You've probably heard it most often in the context of web services ("the Twitter API," "REST APIs"), but the term is broader than that. An API is just the **public face** of any piece of software — the set of things that another programmer is supposed to call when they use your code. It's the contract you offer: "if you use my class this way, here's what it'll do for you."

For a Python class, the API is:
- The class name
- What you pass to its constructor
- What public methods it has (and what they're called)
- What arguments those methods take and what they return
- What guarantees those methods make (e.g., "this returns a tuple of three NumPy arrays of shape `(n_samples,)`")

The API is *not* the internals — not which optimizer you used, not how you loop over the ensemble members, not which intermediate variables you named. Those are implementation details and can change later. The API, once people are depending on it, is much harder to change.

That's why we're designing the API first, **before writing any of the actual ML code**. Three reasons:

1. **It forces you to think about the user, not the algorithm.** What does someone who wants deep ensembles for their problem actually want to call? They don't want to think about training loops; they want `fit()` and `predict_with_uncertainty()`. If you start with the algorithm, you end up exposing the algorithm. If you start with the user, you end up exposing what's useful.
2. **It surfaces design questions early, when they're quicker to fix.** "Wait, should `predict_with_uncertainty` return epistemic and aleatoric variance separately or combined?" is a 10-minute conversation if you have it now and a 3-day refactor if you have it after you've written 800 lines of code.
3. **It gives the team a contract to work toward.** Once the API is agreed, two people can implement two different methods in parallel without stepping on each other, because they both know what their methods need to look like from the outside.

The other deliverable for week 1 is what I'm calling an **API skeleton**: a Python file with the class defined, all the public methods present but with empty bodies, and **real docstrings** explaining what each method will do, what it takes, and what it returns. No actual machine learning yet. Just the shape of the thing.

A classic toy example of what "API skeleton" looks like, for weather:

```python
class WeatherForecaster:
    """Predicts weather for a given location and time window.

    Parameters
    ----------
    location : tuple of (float, float)
        (latitude, longitude) in decimal degrees.
    model_type : {'persistence', 'arima', 'lstm'}, default='arima'
        Which underlying model to use.
    """

    def __init__(self, location, model_type='arima'):
        pass  # implementation comes later

    def fit(self, historical_data):
        """Train the forecaster on historical weather observations.

        Parameters
        ----------
        historical_data : pd.DataFrame
            Must have columns ['timestamp', 'temp', 'humidity', 'pressure'].

        Returns
        -------
        self
        """
        pass

    def predict(self, hours_ahead):
        """Forecast weather for the next N hours.

        Parameters
        ----------
        hours_ahead : int
            Forecast horizon, must be positive.

        Returns
        -------
        forecast : pd.DataFrame
            Columns ['timestamp', 'temp_mean', 'temp_std'].
        """
        pass
```

That's it. No real code. But anyone can read this and tell you exactly how the class is meant to be used. **That's the goal for the end of week 1.**

---

### Team 2 (Deep Ensembles)

- Walk through Tyler's spring code together. Ideally, I'd have asked Tyler to record a walkthrough, but instead, I'll link the junior design presentations from April (note, the first 20 minutes of the video is silent!). The point isn't to necessarily keep that code — it's to learn what worked and what may have been hacky so the class redesign is informed.
- Sketch the API on a shared doc before writing any code. Something scikit-learn shaped:
  ```python
  ensemble = DeepEnsemble(n_models=10, architecture=..., heteroscedastic=False)
  ensemble.fit(X_train, y_train)
  mean, epistemic_var, aleatoric_var = ensemble.predict_with_uncertainty(X_test)
  ensemble.plot_parity_with_uncertainty(X_test, y_test)
  ```
  That's just a starting point — argue with it. Maybe `n_models` should default to something. Maybe `heteroscedastic=False` should be in `fit()` instead of `__init__`. Maybe the plotting belongs in a separate module. These are the kinds of decisions you're making this week.
- Deliverable by end of week 1: a documented API skeleton — empty methods with real docstrings explaining what each will do. No implementation yet. This is the architecture step.

### Team 3 (Bias & Variance Reduction)

- Dayton, lead a debrief on what the spring team got right and wrong. The three-experiment structure from Majdi's spec (model bias, sampling bias, data bias) maps cleanly onto three methods of a class — that's your starting point.
- Sketch the API on a shared doc before writing any code. Something like:
  ```python
  analyzer = BiasAnalyzer(base_architecture=..., dataset=...)
  analyzer.run_model_bias_study(architectures=[...])
  analyzer.run_sampling_bias_study(strategies=['bootstrap','lhs','stratified'])
  analyzer.run_data_bias_study(cv_folds=5)
  analyzer.decompose_variance()
  analyzer.plot_disagreement_map()
  ```
  Same as Team 2 — this is a starting point, not gospel. Push back on it. Is `BiasAnalyzer` even the right name? Should the three studies share state, or be independent? Does `decompose_variance()` need to be called after all three, or can it work after just one? Try to tease these out this week.
- Same week-1 deliverable: a documented API skeleton with empty methods and real docstrings.

## Some things I need to pin down with Majdi at the first check-in

A couple of open questions that affect how we design these classes:

- Whether these classes should follow scikit-learn's `BaseEstimator` conventions exactly, so they can plug into pyMAISE down the road
- Whether there are reference visualizations he wants matched (parity plots with error bars, calibration plots, etc.)
- Documentation expectations — NumPy-style docstrings? A Sphinx site? Just READMEs plus example notebooks?

## Ground rules

A few things I want to be explicit about:

- **Ask questions early and in writing.** Async only works if people actually post. I'd rather see a zillion questions in `#general` than one private DM at week 6 when you've been stuck for a month.
- **Show your work.** Push work in progress (WIP) branches. Open draft PRs. Let people see what you're building before it's "done."
- **No silent weeks.** If I haven't seen a status update or a commit from you in seven days, I'm checking in. Don't ghost. It's ok that some weeks are more productive than others, but ghosting isn't cool!
- **This is real work, not a class.** That's why y'all are getting paid! Majdi and his group are real collaborators expecting real output -- this is exciting for FLC and can help your future!

I'm excited for this summer. We've got a real chance to put out something genuinely useful, and I think y'all are a great crew.

Post questions in `#general`. First all-hands Teams meeting will be on the calendar shortly.

— Matt