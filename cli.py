#!/usr/bin/env python3
"""
Echo Ridge Scoring CLI

Command-line interface for the Echo Ridge deterministic scoring engine.
Provides batch processing, context management, and validation commands.
"""

import sys
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.batch import BatchProcessorCLI, BatchProcessingError


# Initialize Typer app and Rich console
app = typer.Typer(
    name="echo-ridge",
    help="Echo Ridge AI-Readiness Scoring Engine CLI",
    add_completion=False
)
console = Console()


@app.command()
def score(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input JSONL file with company data"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSONL file for scoring results"),
    norm_context: Optional[str] = typer.Option(None, "--norm-context", "-n", help="Specific NormContext version to use"),
    no_db: bool = typer.Option(False, "--no-db", help="Skip database writes (faster, no persistence)"),
    database_url: str = typer.Option("sqlite:///echo_ridge_scoring.db", "--db-url", help="Database URL for persistence"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Score companies from JSONL file with deterministic outputs.
    
    The input file should contain one JSON object per line, each representing
    a company with all required fields (digital, ops, info_flow, market, budget, meta).
    
    Example:
        echo-ridge score -i companies.jsonl -o results.jsonl
        echo-ridge score -i data.jsonl -o output.jsonl --norm-context v1.0 --no-db
    """
    try:
        if verbose:
            console.print(f"[blue]Starting batch scoring...[/blue]")
            console.print(f"Input: {input_file}")
            console.print(f"Output: {output_file}")
            console.print(f"NormContext: {norm_context or 'auto (latest)'}")
            console.print(f"Database writes: {'disabled' if no_db else 'enabled'}")
        
        # Initialize CLI processor
        cli = BatchProcessorCLI(database_url=database_url)
        
        # Process batch
        result = cli.score_batch(
            str(input_file), 
            str(output_file), 
            norm_context, 
            no_db_write=no_db
        )
        
        # Display results
        console.print("\n[green]✓ Batch processing completed successfully![/green]")
        
        # Create summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Companies Processed", str(result["companies_processed"]))
        table.add_row("Successful", str(result["companies_succeeded"]))
        table.add_row("Failed", str(result["companies_failed"]))
        table.add_row("Success Rate", f"{result['success_rate_pct']:.1f}%")
        table.add_row("Processing Time", f"{result['processing_time_ms']:.0f} ms")
        table.add_row("NormContext Version", result["norm_context_version"])
        
        if result.get("batch_id"):
            table.add_row("Batch ID", result["batch_id"])
        
        console.print(table)
        
        # Show errors if any
        if result["error_count"] > 0:
            console.print(f"\n[yellow]⚠️  {result['error_count']} errors occurred during processing[/yellow]")
            if verbose and result.get("errors"):
                for error in result["errors"][:5]:  # Show first 5 errors
                    console.print(f"  Line {error['line_number']}: {error['error']}")
                if result["error_count"] > 5:
                    console.print(f"  ... and {result['error_count'] - 5} more errors")
        
    except BatchProcessingError as e:
        console.print(f"[red]✗ Batch processing failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def validate(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input JSONL file to validate"),
    norm_context: Optional[str] = typer.Option(None, "--norm-context", "-n", help="NormContext version to use"),
    database_url: str = typer.Option("sqlite:///echo_ridge_scoring.db", "--db-url", help="Database URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Validate deterministic processing by running the same input twice.
    
    This command processes the input file twice with identical parameters
    and verifies that the outputs are identical (same checksum).
    
    Example:
        echo-ridge validate -i test_data.jsonl
        echo-ridge validate -i companies.jsonl --norm-context v1.0
    """
    try:
        if verbose:
            console.print(f"[blue]Validating deterministic processing...[/blue]")
            console.print(f"Input file: {input_file}")
        
        # Initialize CLI processor
        cli = BatchProcessorCLI(database_url=database_url)
        
        # Run validation
        result = cli.validate_deterministic(str(input_file), norm_context)
        
        # Display results
        if result["is_reproducible"]:
            console.print("\n[green]✓ Processing is deterministic and reproducible![/green]")
        else:
            console.print("\n[red]✗ Processing is NOT reproducible![/red]")
        
        # Create detailed table
        table = Table(title="Reproducibility Validation")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Is Reproducible", "✓ Yes" if result["is_reproducible"] else "✗ No")
        table.add_row("Companies Processed", str(result["companies_processed"]))
        table.add_row("NormContext Version", result["norm_context_version"])
        table.add_row("First Run Checksum", result["checksum1"][:16] + "...")
        table.add_row("Second Run Checksum", result["checksum2"][:16] + "...")
        
        console.print(table)
        
        if verbose:
            console.print("\n[dim]First run summary:[/dim]")
            console.print(json.dumps(result["result1"], indent=2))
            console.print("\n[dim]Second run summary:[/dim]")  
            console.print(json.dumps(result["result2"], indent=2))
        
        if not result["is_reproducible"]:
            raise typer.Exit(1)
            
    except BatchProcessingError as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def contexts(
    database_url: str = typer.Option("sqlite:///echo_ridge_scoring.db", "--db-url", help="Database URL")
):
    """
    List available NormContext versions.
    
    Shows all stored normalization contexts with their metadata.
    
    Example:
        echo-ridge contexts
    """
    try:
        cli = BatchProcessorCLI(database_url=database_url)
        contexts = cli.list_contexts()
        
        if not contexts:
            console.print("[yellow]No NormContext versions found.[/yellow]")
            console.print("Run 'echo-ridge score' to create one automatically.")
            return
        
        table = Table(title="Available NormContext Versions")
        table.add_column("Version", style="cyan")
        table.add_column("Status", style="magenta")
        
        for ctx in contexts:
            table.add_row(ctx["version"], ctx["status"])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Error listing contexts: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def batch_status(
    batch_id: str = typer.Argument(..., help="Batch ID to check"),
    database_url: str = typer.Option("sqlite:///echo_ridge_scoring.db", "--db-url", help="Database URL")
):
    """
    Get status and summary for a specific batch run.
    
    Example:
        echo-ridge batch-status abc123def456
    """
    try:
        cli = BatchProcessorCLI(database_url=database_url)
        status = cli.get_batch_status(batch_id)
        
        if status is None:
            console.print(f"[red]✗ Batch ID '{batch_id}' not found[/red]")
            raise typer.Exit(1)
        
        # Display batch information
        panel = Panel(
            f"[bold]Batch ID:[/bold] {status['batch_id']}\n"
            f"[bold]Status:[/bold] Completed\n"
            f"[bold]Processing Time:[/bold] {status['processing_time_ms']:.0f} ms\n"
            f"[bold]Success Rate:[/bold] {status['success_rate']:.1%}",
            title="Batch Run Summary"
        )
        console.print(panel)
        
        # Create detailed table
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Input File", status["input_file_path"])
        table.add_row("Output File", status["output_file_path"])
        table.add_row("NormContext Version", status["norm_context_version"])
        table.add_row("Companies Processed", str(status["companies_processed"]))
        table.add_row("Successful", str(status["companies_succeeded"]))
        table.add_row("Failed", str(status["companies_failed"]))
        table.add_row("Started At", status["started_at"])
        table.add_row("Completed At", status["completed_at"])
        table.add_row("Input Checksum", status["input_checksum"][:16] + "...")
        table.add_row("Output Checksum", status["output_checksum"][:16] + "...")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Error retrieving batch status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show Echo Ridge scoring engine version."""
    console.print("[bold blue]Echo Ridge Scoring Engine[/bold blue]")
    console.print("Version: 2.0.0 (Phase 4-10 Implementation)")
    console.print("Features: Risk Assessment, Feasibility Gates, Batch Processing")


@app.callback()
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """
    Echo Ridge AI-Readiness Scoring Engine CLI
    
    A deterministic, reproducible scoring system for evaluating SMB companies
    for AI implementation readiness. Provides risk assessment, feasibility gates,
    and batch processing capabilities.
    """
    if version_flag:
        version()
        raise typer.Exit()


if __name__ == "__main__":
    app()