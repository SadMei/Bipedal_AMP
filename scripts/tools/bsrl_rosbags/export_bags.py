#!/usr/bin/env python3
"""Offline ROS1 bag extraction. Original bags and robot settings are never edited.

Run with the isaaclab Python environment (rosbags + numpy are already installed).
Numeric data are retained at recorded rate, with bag/header/monotonic timestamps.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

if __package__:
    from ._paths import BAG_ROOT, RESULTS_ROOT
else:
    from _paths import BAG_ROOT, RESULTS_ROOT


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def extract_values(msg, topic):
    if topic in ("/bsrl/realtime_state", "/bsrl/policy_trace"):
        return {f.name: getattr(msg, f.name) for f in fields(msg)
                if f.name not in ("header", "session_id", "__msgtype__")}
    if topic == "/imu/data":
        return {
            "quaternion_xyzw": np.array([getattr(msg.orientation, k) for k in "xyzw"]),
            "angular_velocity": np.array([getattr(msg.angular_velocity, k) for k in "xyz"]),
            "linear_acceleration": np.array([getattr(msg.linear_acceleration, k) for k in "xyz"]),
            "orientation_covariance": msg.orientation_covariance,
            "angular_velocity_covariance": msg.angular_velocity_covariance,
            "linear_acceleration_covariance": msg.linear_acceleration_covariance,
        }
    if topic == "/cmd_vel":
        return {"velocity": np.array([msg.linear.x, msg.linear.y, msg.angular.z])}
    raise ValueError(topic)


def extract(path, output):
    dest = output / path.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    names = {"/bsrl/realtime_state": "state", "/bsrl/policy_trace": "policy",
             "/imu/data": "imu", "/cmd_vel": "command"}
    with AnyReader([path]) as reader:
        meta = dict(bag=str(path.resolve()), duration_s=reader.duration / 1e9,
                    message_count=reader.message_count, start_time_ns=reader.start_time,
                    topics={})
        counts = {}
        for conn in reader.connections:
            counts[conn.topic] = counts.get(conn.topic, 0) + conn.msgcount
            meta["topics"][conn.topic] = {"type": conn.msgtype, "count": counts[conn.topic]}
        arrays = {}
        indices = {topic: 0 for topic in names}
        health, events = [], []
        for conn, timestamp, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic in names:
                values = extract_values(msg, conn.topic)
                if conn.topic not in arrays:
                    arrays[conn.topic] = {
                        key: np.empty((counts[conn.topic],) + np.asarray(value).shape,
                                      dtype=np.asarray(value).dtype)
                        for key, value in values.items()}
                    arrays[conn.topic]["bag_time_ns"] = np.empty(counts[conn.topic], dtype=np.int64)
                    if hasattr(msg, "header"):
                        arrays[conn.topic]["header_time_ns"] = np.empty(counts[conn.topic], dtype=np.int64)
                data = arrays[conn.topic]
                i = indices[conn.topic]
                for key, value in values.items():
                    data[key][i] = value
                data["bag_time_ns"][i] = timestamp
                if hasattr(msg, "header"):
                    data["header_time_ns"][i] = stamp_ns(msg)
                indices[conn.topic] += 1
            elif conn.topic == "/bsrl/realtime_health":
                health.append({"bag_time_ns": timestamp, **json.loads(msg.data)})
            elif conn.topic == "/bsrl/realtime_event":
                events.append(dict(bag_time_ns=timestamp, header_time_ns=stamp_ns(msg),
                                   event=msg.event, detail=msg.detail,
                                   cycle_sequence=msg.cycle_sequence,
                                   configuration_generation=msg.configuration_generation))
        for topic, data in arrays.items():
            assert indices[topic] == counts[topic]
            np.savez_compressed(dest / f"{names[topic]}.npz", **data)
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (dest / "health.json").write_text(json.dumps(health, indent=2) + "\n")
        (dest / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    print(f"Extracted {path.parent.name}: {indices}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BAG_ROOT)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    for path in sorted(args.root.glob("*/control.bag")):
        extract(path, args.output)


if __name__ == "__main__":
    main()
