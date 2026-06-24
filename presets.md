Key preset behaviors found in codebase:

Effort	AQ (Adaptive QP)	Preanalysis Steering	QP-Aware Pruning	Parallel
Floor	Yes (non-reference)	No	No	Yes
FloorPlus	Yes	No	No	Yes
FloorPlus2	Yes	No	No	Yes
FloorShallow	Yes	No	No	Yes
Fastest	Yes	Map-only (AQ)	Yes	Yes
Fast	Yes	Yes	Yes	Yes
Balanced	Yes	Yes	Yes	Yes
Good	Yes	Yes (extra candidates)	Yes	Yes
Best	No (uniform QP)	No (inert)	No (exhaustive)	Yes (WPP)
Placebo	No (reference)	No (inert)	No	Yes (wavefront)
Reference	No (reference)	No (inert)	No	No (serial