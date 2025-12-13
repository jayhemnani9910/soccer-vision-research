#!/usr/bin/env python3
"""
Query script for the video analytics database.
Allows you to ask questions about analyzed videos.
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

DB_PATH = Path("vlm_outputs/video_analytics.db")

def query_objects_by_name(name: str, limit: int = 10) -> List[Dict]:
    """Find all frames containing a specific object."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
    SELECT f.frame_id, f.scene_id, f.timestamp_s, f.image_path, f.scanner_json
    FROM frames f
    WHERE f.scanner_json LIKE ?
    ORDER BY f.timestamp_s
    LIMIT ?
    """, (f'%"{name}"%', limit))
    
    results = []
    for row in cur.fetchall():
        scanner = json.loads(row['scanner_json'])
        objects = [obj for obj in scanner.get('objects', []) if name.lower() in obj.get('name', '').lower()]
        results.append({
            'frame_id': row['frame_id'],
            'scene_id': row['scene_id'],
            'timestamp_s': row['timestamp_s'],
            'image_path': row['image_path'],
            'objects': objects
        })
    
    conn.close()
    return results

def query_text_content(text: str, limit: int = 10) -> List[Dict]:
    """Find all frames containing specific text."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
    SELECT f.frame_id, f.scene_id, f.timestamp_s, f.image_path, f.scanner_json
    FROM frames f
    WHERE f.scanner_json LIKE ?
    ORDER BY f.timestamp_s
    LIMIT ?
    """, (f'%{text}%', limit))
    
    results = []
    for row in cur.fetchall():
        scanner = json.loads(row['scanner_json'])
        text_spotted = [t for t in scanner.get('text_spotted', []) if text.lower() in t.get('content', '').lower()]
        results.append({
            'frame_id': row['frame_id'],
            'scene_id': row['scene_id'],
            'timestamp_s': row['timestamp_s'],
            'image_path': row['image_path'],
            'text_found': text_spotted
        })
    
    conn.close()
    return results

def query_events_by_type(event_type: str, limit: int = 10) -> List[Dict]:
    """Find all events of a specific type."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
    SELECT e.event_id, e.scene_id, e.frame_id, f.timestamp_s, e.type, e.description, e.payload
    FROM events e
    LEFT JOIN frames f ON e.frame_id = f.frame_id
    WHERE e.type LIKE ?
    ORDER BY f.timestamp_s
    LIMIT ?
    """, (f'%{event_type}%', limit))
    
    results = []
    for row in cur.fetchall():
        results.append({
            'event_id': row['event_id'],
            'scene_id': row['scene_id'],
            'frame_id': row['frame_id'],
            'timestamp_s': row['timestamp_s'],
            'type': row['type'],
            'description': row['description'],
            'payload': json.loads(row['payload']) if row['payload'] else None
        })
    
    conn.close()
    return results

def get_scene_summary(scene_id: int) -> Dict:
    """Get detailed information about a specific scene."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get frames in this scene
    cur.execute("""
    SELECT f.frame_id, f.timestamp_s, f.scanner_json, f.tracker_json
    FROM frames f
    WHERE f.scene_id = ?
    ORDER BY f.timestamp_s
    """, (scene_id,))
    
    frames = []
    for row in cur.fetchall():
        frames.append({
            'frame_id': row['frame_id'],
            'timestamp_s': row['timestamp_s'],
            'scanner': json.loads(row['scanner_json']),
            'tracker': json.loads(row['tracker_json'])
        })
    
    # Get events in this scene
    cur.execute("""
    SELECT e.event_id, e.frame_id, e.type, e.description, e.payload
    FROM events e
    WHERE e.scene_id = ?
    ORDER BY e.frame_id
    """, (scene_id,))
    
    events = []
    for row in cur.fetchall():
        events.append({
            'event_id': row['event_id'],
            'frame_id': row['frame_id'],
            'type': row['type'],
            'description': row['description'],
            'payload': json.loads(row['payload']) if row['payload'] else None
        })
    
    # Get QA questions for this scene
    cur.execute("""
    SELECT qa.qa_id, qa.frame_id, qa.question, qa.qtype, qa.hint
    FROM qa_questions qa
    WHERE qa.scene_id = ?
    ORDER BY qa.frame_id
    """, (scene_id,))
    
    questions = []
    for row in cur.fetchall():
        questions.append({
            'qa_id': row['qa_id'],
            'frame_id': row['frame_id'],
            'question': row['question'],
            'type': row['qtype'],
            'hint': row['hint']
        })
    
    conn.close()
    
    return {
        'scene_id': scene_id,
        'frames': frames,
        'events': events,
        'questions': questions
    }

def list_all_scenes() -> List[Dict]:
    """List all scenes with basic info."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
    SELECT 
        s.scene_id,
        MIN(f.timestamp_s) as start_time,
        MAX(f.timestamp_s) as end_time,
        COUNT(f.frame_id) as frame_count
    FROM scenes s
    LEFT JOIN frames f ON s.scene_id = f.scene_id
    GROUP BY s.scene_id
    ORDER BY s.scene_id
    """)
    
    scenes = []
    for row in cur.fetchall():
        scenes.append({
            'scene_id': row['scene_id'],
            'start_time_s': row['start_time'],
            'end_time_s': row['end_time'],
            'duration_s': row['end_time'] - row['start_time'] if row['start_time'] and row['end_time'] else 0,
            'frame_count': row['frame_count']
        })
    
    conn.close()
    return scenes

def main():
    parser = argparse.ArgumentParser(description='Query video analytics database')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Objects query
    obj_parser = subparsers.add_parser('objects', help='Find objects by name')
    obj_parser.add_argument('name', help='Object name to search for')
    obj_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Text query
    text_parser = subparsers.add_parser('text', help='Find frames containing text')
    text_parser.add_argument('text', help='Text content to search for')
    text_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Events query
    event_parser = subparsers.add_parser('events', help='Find events by type')
    event_parser.add_argument('type', help='Event type to search for')
    event_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Scene summary
    scene_parser = subparsers.add_parser('scene', help='Get scene details')
    scene_parser.add_argument('scene_id', type=int, help='Scene ID')
    
    # List scenes
    subparsers.add_parser('scenes', help='List all scenes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run analyze_youtube_vlm.py first to create the database.")
        return
    
    if args.command == 'objects':
        results = query_objects_by_name(args.name, args.limit)
        print(f"\nFound {len(results)} frames with object '{args.name}':")
        for r in results:
            print(f"  Scene {r['scene_id']}, Frame {r['frame_id']}, t={r['timestamp_s']:.2f}s")
            for obj in r['objects']:
                print(f"    - {obj.get('name', 'Unknown')}: {obj.get('attributes', {})}")
    
    elif args.command == 'text':
        results = query_text_content(args.text, args.limit)
        print(f"\nFound {len(results)} frames with text '{args.text}':")
        for r in results:
            print(f"  Scene {r['scene_id']}, Frame {r['frame_id']}, t={r['timestamp_s']:.2f}s")
            for t in r['text_found']:
                print(f"    - '{t.get('content', '')}'")
    
    elif args.command == 'events':
        results = query_events_by_type(args.type, args.limit)
        print(f"\nFound {len(results)} events of type '{args.type}':")
        for r in results:
            print(f"  Scene {r['scene_id']}, Frame {r['frame_id']}, t={r['timestamp_s']:.2f}s")
            print(f"    - {r['description']}")
    
    elif args.command == 'scene':
        scene = get_scene_summary(args.scene_id)
        print(f"\nScene {scene['scene_id']} Summary:")
        print(f"  Frames: {len(scene['frames'])}")
        print(f"  Events: {len(scene['events'])}")
        print(f"  Questions: {len(scene['questions'])}")
        
        if scene['frames']:
            first_frame = scene['frames'][0]
            last_frame = scene['frames'][-1]
            print(f"  Time range: {first_frame['timestamp_s']:.2f}s - {last_frame['timestamp_s']:.2f}s")
            
            print("\n  Frame summaries:")
            for f in scene['frames']:
                summary = f['scanner'].get('scene_summary', 'No summary')
                print(f"    t={f['timestamp_s']:.2f}s: {summary}")
    
    elif args.command == 'scenes':
        scenes = list_all_scenes()
        print(f"\nAll scenes ({len(scenes)} total):")
        for s in scenes:
            print(f"  Scene {s['scene_id']}: {s['duration_s']:.2f}s, {s['frame_count']} frames")

if __name__ == "__main__":
    main()