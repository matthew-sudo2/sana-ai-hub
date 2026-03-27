#!/usr/bin/env python3
"""
Backup Cleanup Utility: Archives old model backups
Keeps only the N most recent backups in models/, moves older ones to archived/backups/

Usage:
    python cleanup_backups.py                 # Default: keep 3 recent
    python cleanup_backups.py --keep 5       # Keep 5 most recent
    python cleanup_backups.py --dry-run      # Show what would be archived
"""

from pathlib import Path
from datetime import datetime
import shutil
import sys
import argparse


def cleanup_backups(models_dir: str = "models", keep: int = 3, dry_run: bool = False):
    """
    Archive old backups, keeping only N most recent.
    
    Args:
        models_dir: Path to models directory
        keep: Number of recent backups to keep in models/
        dry_run: If True, show what would be archived without doing it
    """
    models_path = Path(models_dir)
    archive_backups_dir = models_path / "archived" / "backups"
    
    # Find all backup files
    backup_files = sorted(models_path.glob("best_model_bak_*.pkl"))
    
    if not backup_files:
        print("✓ No backups to clean up")
        return 0
    
    print("="*80)
    print("MODEL BACKUP CLEANUP UTILITY")
    print("="*80)
    print(f"\nFound {len(backup_files)} backup(s)")
    print(f"Policy: Keep {keep} most recent, archive older\n")
    
    if len(backup_files) <= keep:
        print(f"✓ All {len(backup_files)} backup(s) are recent enough (threshold: {keep})")
        print("  No archival needed.\n")
        return 0
    
    # Determine which to archive
    to_keep = backup_files[-keep:]
    to_archive = backup_files[:-keep]
    
    print(f"ACTION: Archive {len(to_archive)} old backup(s)")
    print(f"        Keep {len(to_keep)} recent backup(s)\n")
    
    # Show what will happen
    print("RECENT BACKUPS (keeping in models/):")
    for f in to_keep:
        print(f"  ✓ {f.name}")
    
    print("\nOLD BACKUPS (archiving to models/archived/backups/):")
    for f in to_archive:
        # Extract timestamp from filename
        # Format: best_model_bak_20260327_061432.pkl
        timestamp_str = f.name.replace("best_model_bak_", "").replace(".pkl", "")
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            age_str = f" (from {dt.strftime('%Y-%m-%d %H:%M:%S')})"
        except:
            age_str = ""
        print(f"  ✗ {f.name}{age_str}")
    
    if dry_run:
        print("\n[DRY RUN] No files were actually moved.\n")
        return 0
    
    # Create archive directory
    archive_backups_dir.mkdir(parents=True, exist_ok=True)
    
    # Move old backups
    print("\n" + "-"*80)
    print("ARCHIVING...\n")
    
    archived_count = 0
    for old_file in to_archive:
        try:
            dest = archive_backups_dir / old_file.name
            
            # Handle if file already exists in archive
            if dest.exists():
                # Rename to add timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = archive_backups_dir / f"{old_file.stem}_archived_{timestamp}.pkl"
            
            shutil.move(str(old_file), str(dest))
            print(f"  ✓ Archived: {old_file.name}")
            archived_count += 1
        except Exception as e:
            print(f"  ✗ Failed to archive {old_file.name}: {e}")
    
    print("\n" + "="*80)
    print(f"✓ CLEANUP COMPLETE: Archived {archived_count}/{len(to_archive)} old backup(s)")
    print("="*80 + "\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Archive old model backups, keeping only recent ones"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of recent backups to keep (default: 3)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to models directory (default: models)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without doing it"
    )
    
    args = parser.parse_args()
    
    return cleanup_backups(
        models_dir=args.models_dir,
        keep=args.keep,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
