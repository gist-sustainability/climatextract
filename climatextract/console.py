"""Rich-based console output module for ClimXtract CLI."""

import logging
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from rich.console import Console as RichConsole
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# Single source of truth: read version from __init__.py at runtime
# (avoids circular import since we only read __version__ at first use via print_header)
_VERSION = None

def _get_version() -> str:
    global _VERSION
    if _VERSION is None:
        try:
            from climatextract import __version__
            _VERSION = __version__
        except ImportError:
            _VERSION = "unknown"
    return _VERSION


def _suppress_third_party_logs():
    """Suppress noisy third-party loggers.

    Single source of truth for all logger suppression across ClimXtract.
    Called once during console initialization.
    """
    noisy_loggers = [
        "azure.core",
        "azure.identity",
        "azure.storage",
        "httpx",
        "httpcore",
        "urllib3",
        "urllib3.connectionpool",
        "openai",
        "llama_index",
        "mlflow",
        "mlflow.tracing",
        "mlflow.tracing.processor",
        "mlflow.tracing.processor.mlflow",
        "docling",
        "docling_core",
        "transformers",
        "torch",
        "pdf2image",
        "PIL",
        "timm",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Suppress RuntimeWarning about module imports
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib")
    warnings.filterwarnings("ignore", message=".*module.*was already imported.*")


@dataclass
class ProgressTracker:
    """Tracks progress state across extraction steps."""
    total_pdfs: int = 0
    pdfs_needing_embedding: int = 0
    pdfs_embedded: int = 0
    pdfs_extracted: int = 0
    warnings: List[str] = None
    encrypted_pdfs: int = 0
    failed_pdfs: int = 0
    has_unresolved_duplicates: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class Console:
    """Rich-based console for ClimXtract CLI output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._console = RichConsole()
        self.tracker = ProgressTracker()
        self._progress = None
        self._embedding_task = None
        self._extraction_task = None

    def print_header(self):
        """Print the ClimXtract header banner."""
        header_text = f"  ClimXtract v{_get_version()} - CO2 Emissions Extraction"
        self._console.print(Panel(header_text, box=box.ROUNDED, padding=(0, 2)))
        self._console.print()

    def print_config(self, llm_model: str, embedding_model: str, input_mode: str, pdf_count: int):
        """Print configuration section."""
        self._console.print("[bold]Configuration[/bold]")
        self._console.print(f"  ├─ LLM Model:       {llm_model}")
        self._console.print(f"  ├─ Embedding Model: {embedding_model}")
        self._console.print(f"  ├─ Input Mode:      {input_mode}")
        self._console.print(f"  └─ PDFs to process: {pdf_count}")
        self._console.print()

        self.tracker.total_pdfs = pdf_count

    def print_setup(self, database_name: str, database_exists: bool,
                   pdfs_needing_embedding: int, total_pdfs: int,
                   search_query_cached: bool = True,
                   pdf_cache_status: Optional[Dict[str, bool]] = None):
        """Print setup section showing database and embedding status."""
        self._console.print("[bold]Setup[/bold]")

        # Database status
        db_display = database_name.split("/")[-1] if "/" in database_name else database_name
        if database_exists:
            self._console.print(f"  ├─ Database: {db_display}")
        else:
            self._console.print(f"  ├─ Database: {db_display} [dim](not found)[/dim]")
            self._console.print(f"  ├─ Creating new database...")

        # Verbose mode: show search query and per-PDF status
        if self.verbose and pdf_cache_status:
            query_status = "cached ✓" if search_query_cached else "new"
            self._console.print(f"  ├─ Search query: {query_status}")

            for pdf_name, cached in pdf_cache_status.items():
                status = "cached" if cached else "needs embedding"
                self._console.print(f"  ├─ {pdf_name}: {status}")

        # Embedding status summary
        self.tracker.pdfs_needing_embedding = pdfs_needing_embedding
        if pdfs_needing_embedding == 0:
            self._console.print(f"  └─ All embeddings cached ✓")
        elif pdfs_needing_embedding == total_pdfs and not database_exists:
            self._console.print(f"  └─ All {total_pdfs} PDFs need embedding")
        else:
            self._console.print(f"  └─ {pdfs_needing_embedding} of {total_pdfs} PDFs need embedding")

        self._console.print()

    def _ensure_progress_started(self):
        """Create and start the shared Progress instance if it doesn't exist."""
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=self._console,
                transient=not self.verbose
            )
            self._progress.start()

    def start_embedding_progress(self, total: int):
        """Start the embedding progress bar."""
        if total == 0:
            return
        self._ensure_progress_started()
        self._embedding_task = self._progress.add_task(
            "[bold blue]Embedding[/bold blue]", total=total)

    def update_embedding_progress(self, pdf_name: Optional[str] = None, pages: int = 0):
        """Update embedding progress."""
        if self._progress and self._embedding_task is not None:
            self._progress.update(self._embedding_task, advance=1)
            self.tracker.pdfs_embedded += 1

            if self.verbose and pdf_name:
                self._console.print(f"  [bold blue]Embedding[/bold blue] [{self.tracker.pdfs_embedded}/{self.tracker.pdfs_needing_embedding}] {pdf_name} → {pages} pages ✓")

    def stop_embedding_progress(self):
        """Stop the embedding progress bar."""
        if self._progress and self._embedding_task is not None:
            self._progress.remove_task(self._embedding_task)
            self._embedding_task = None
            if self._extraction_task is None:
                self._progress.stop()
                self._progress = None

    def start_extraction_progress(self, total: int):
        """Start the extraction progress bar."""
        self._ensure_progress_started()
        self._extraction_task = self._progress.add_task(
            "[bold green]Extracting[/bold green]", total=total)

    def update_extraction_progress(self, pdf_name: Optional[str] = None,
                                   pages: int = 0, tables: int = 0,
                                   values: int = 0):
        """Update extraction progress."""
        if self._progress and self._extraction_task is not None:
            self._progress.update(self._extraction_task, advance=1)
            self.tracker.pdfs_extracted += 1

            if self.verbose and pdf_name:
                detail_parts = [f"Pages: {pages}"]
                if tables > 0:
                    detail_parts.append(f"Pages w/ tables: {tables}")
                detail_parts.append(f"Values: {values} ✓")
                details = " | ".join(detail_parts)

                self._console.print(f"\n  [bold green]Extracting[/bold green] [{self.tracker.pdfs_extracted}/{self.tracker.total_pdfs}] {pdf_name}")
                self._console.print(f"        {details}")

    def stop_extraction_progress(self):
        """Stop the extraction progress bar."""
        if self._progress and self._extraction_task is not None:
            self._progress.remove_task(self._extraction_task)
            self._extraction_task = None
            if self._embedding_task is None:
                self._progress.stop()
                self._progress = None
            # In normal mode, transient progress bar vanishes — print a summary
            if not self.verbose and self.tracker.pdfs_extracted > 0:
                self._console.print(
                    f"  Extracted {self.tracker.pdfs_extracted} of {self.tracker.total_pdfs} PDFs")
            self._console.print()

    def print_evaluation(self, metrics: Dict[str, Any],
                        processed_count: int,
                        gold_standard_count: int):
        """Print evaluation results."""
        self._console.print("[bold]Evaluation[/bold]")

        self._console.print(f"  ✓ Compared {processed_count} of {gold_standard_count} gold standard reports")

        precision = metrics.get("precision_value", 0)
        recall = metrics.get("recall_value", 0)
        f1 = metrics.get("f1_value", 0)

        self._console.print(f"  ├─ Precision: {precision:.2f}")
        self._console.print(f"  ├─ Recall:    {recall:.2f}")
        self._console.print(f"  └─ F1 Score:  {f1:.2f}")

        if self.verbose:
            self._console.print("  │")
            self._console.print(f"  ├─ Gold standard total: {gold_standard_count} reports")
            self._console.print(f"  ├─ Processed:           {processed_count} reports")
            not_processed = gold_standard_count - processed_count
            if not_processed > 0:
                self._console.print(f"  └─ Not in extraction:   {not_processed} reports (skipped)")

        self._console.print()

    def print_warnings(self):
        """Print accumulated warnings."""
        warnings_to_show = []

        if self.tracker.encrypted_pdfs > 0:
            warnings_to_show.append(f"{self.tracker.encrypted_pdfs} PDFs skipped (encrypted)")
        if self.tracker.failed_pdfs > 0:
            warnings_to_show.append(f"{self.tracker.failed_pdfs} PDFs failed to process")
        if self.tracker.has_unresolved_duplicates:
            warnings_to_show.append("Duplicates found for some (Report, Scope, Year) - wide format not generated")

        warnings_to_show.extend(self.tracker.warnings)

        if warnings_to_show:
            self._console.print("[bold yellow]⚠ Warnings[/bold yellow]")
            for warning in warnings_to_show:
                self._console.print(f"  • {warning}")
            self._console.print()

    def print_results_start(self, results_path: str):
        """Print warnings and the top of the results section."""
        self.print_warnings()
        self._console.rule()
        self._console.print(f"📋 Results: {results_path}")

    def print_results_end(self):
        """Print the bottom of the results section."""
        self._console.rule()

    def print_results(self, results_path: str):
        """Print the complete results section (non-MLflow path)."""
        self.print_results_start(results_path)
        self.print_results_end()

    def add_warning(self, warning: str):
        """Add a warning to be displayed at the end."""
        if warning not in self.tracker.warnings:
            self.tracker.warnings.append(warning)

    def record_encrypted_pdf(self, pdf_name: str):
        """Record an encrypted PDF."""
        self.tracker.encrypted_pdfs += 1

    def record_failed_pdf(self, pdf_name: str):
        """Record a failed PDF."""
        self.tracker.failed_pdfs += 1

    def record_unresolved_duplicates(self):
        """Record that unresolved duplicates were found."""
        self.tracker.has_unresolved_duplicates = True


# Global console instance
_console: Optional[Console] = None


def get_console() -> Console:
    """Get the global console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def init_console(verbose: bool = False) -> Console:
    """Initialize the global console with settings."""
    global _console
    _console = Console(verbose=verbose)
    _suppress_third_party_logs()

    # Route all logging through Rich so log messages don't disrupt progress bars
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(RichHandler(
        console=_console._console,
        show_path=False,
        show_time=False,
    ))

    return _console
