> Historical single-seed pilot. The current project is the [four-algorithm, three-seed warehouse comparison](warehouse_comparison_results.md); this earlier run is excluded from that matrix. UI descriptions below refer to its earlier stage.

# Warehouse delivery robot

The default demonstration represents a small warehouse, rather than an unexplained obstacle grid. A delivery robot collects stock at a receiving bay or a storage-rack access point, carries it to packing, moves packed parcels to dispatch, and returns to a charging dock.

## Floor plan and purpose

The new, separately versioned `warehouse-delivery-v1` floor has 13 rows and 17 columns, including its outer walls. Its 130 traversable squares form one connected floor. Three storage racks have regular parallel aisles, with an upper circulation aisle and a main cross-aisle below the racks. Receiving, packing and dispatch occupy labelled areas along the bottom of the floor. A packing bench and inbound pallet staging are explicit obstacles. The charging dock is in the upper perimeter aisle.

Pickup points are on the aisle beside a rack, not inside a blocked shelf. Each job names its pickup point, destination and purpose. The dark dashboard styling is unchanged. Both the 2D floor plan and temporary 3D preview annotate the same fixtures and operational areas.

The seven saved jobs cover stock put-away, order picking, parcel dispatch and returning for charging. Users can also assign a delivery between any two distinct named stations. A new route is generated using the warehouse's trained Q-learning model or the explicitly labelled BFS shortest-route reference.

## The matching learned model

The old coordinate-indexed checkpoints were trained for different geometry and cannot navigate this floor correctly. The warehouse has its own Q-learning table, trained once with seed 19301 for a fixed 600,000 interactions. Its configuration was written before training: learning rate 0.3, discount 0.99, per-move reward -0.05, goal reward +10, collision reward -5, and a 200-move cap. Exploration decreases from 1 to 0.03. A collision remains terminal and time limits retain bootstrap semantics, matching the existing environment.

The training set contains 715 start–goal pairs, with destinations at the seven warehouse stations. The 188 development pairs are disjoint from those training starts, and include every named demonstration job. Trajectories can still visit overlapping states. Named jobs were monitored every 100,000 interactions. The final checkpoint at the fixed budget was retained; no checkpoint selection or extra tuning occurred.

Reloaded-checkpoint validation completed 188/188 development routes, with no collisions or timeouts and mean SPL 0.982807. All seven named jobs completed. These are inspected development results from one training seed on a familiar floor, **not** an independent final assessment, an unseen-warehouse result or a real-robot guarantee. No historical final cohort was rerun.

The map, route split, training settings, output checksums and exact training-engine source are retained in `data/warehouse/`, `configs/warehouse_demo.yaml` and `artifacts/warehouse/warehouse-delivery-q-learning-v1/`. Opening or playing the app does not train the model.

## Historical map and limits

`bottleneck-v1` remains accessible as **Historical benchmark - abstract grid**. It represents an abstract research task, not this warehouse. Its original models, recordings and 90.5% selected-model final result retain their original meaning. Research tests explicitly select that historical map; the everyday app defaults to the warehouse.

The simulation models a point robot, four grid moves and static fixtures. It does not animate grabbing a parcel, model cargo, account for a physical robot's size, detect people, use live sensors or move vertically. Warehouse labels give the navigation task a concrete purpose without claiming those additional capabilities.
