# Digital twin and runtime

The local research prototype includes 3D turbines, wake/component responses, annual measurements, 72-hour forecasts and progressive retrospective truth reveal. Private-data modules are not distributed. This public edition is a standalone synthetic demonstration using Three.js, twelve procedurally generated positions, schematic rotors and analytic weather paths; no real coordinates or telemetry.

Run `python demo/server.py` with Python 3.10+ and open http://127.0.0.1:8788; Windows users may double-click start_demo.cmd. No third-party Python packages are needed. A WebGL browser is required. The server binds only localhost and serves only demo/static. Stop with Ctrl+C. Three.js is bundled for offline use with its MIT notice.

Orbit/zoom, change synthetic wind and task capacity, inspect six-hour probabilities from 57 synthetic paths and a 72-hour mean, or switch languages. Selection uses p≥0.8 and a capacity cap; it is not the full robust optimizer. The public edition reuses model.js/farm.js power interpolation, first-order thermal response and Jensen/Park wake calculations. Layout and power/Ct curves are replaced with fictional analytic inputs. Total power, wake loss and turbine-1 bearing temperature are displayed; rotors follow simulated RPM. Parameters are uncalibrated and do not represent a validated turbine or site.

All interactive outputs are generated synthetically. Published aggregate research tables/figures come from frozen private experiments, not this demo. Archived research code requires separately authorized private inputs. Fitted weights, original scenarios and replay caches are withheld. Screenshots are not evidence of field validation.
