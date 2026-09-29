$root = "D:\research\jet_tagging"
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"
$log = "$root\experiments\physics_sr\stageB_run.log"
Start-Transcript -Path $log -Append
try {
  # 8 screen configs (seed 42 only); each x (z, robust) = 16 cells
  $cfgs = @(
    @("pars1e-5", @("--parsimony", "1e-5", "--no-nest-tanh")),
    @("pars3e-4", @("--parsimony", "3e-4", "--no-nest-tanh")),
    @("pars1e-3", @("--parsimony", "1e-3", "--no-nest-tanh")),
    @("noexplog", @("--opset", "NOEXPLOG", "--no-nest-tanh")),
    @("pluspow",  @("--opset", "PLUSPOW", "--no-nest-tanh")),
    @("tanhonly", @("--opset", "TANHONLY", "--no-nest-tanh")),
    @("nestok",   @()),
    @("ms35",     @("--maxsize", "35", "--no-nest-tanh"))
  )
  foreach ($c in $cfgs) {
    $cname, $extra = $c[0], $c[1]
    foreach ($norm in @("z", "robust")) {
      $tag = "stgB_${cname}_${norm}_s42"
      Write-Output "[stageB] fit start cfg=$cname norm=$norm $(Get-Date -Format o)"
      & $py -m scripts.analysis.fit_physics_sr --mode quick --seed 42 `
          --norm $norm --out-tag $tag @extra
      if ($?) {
        Write-Output "[stageB] eval-self start $tag $(Get-Date -Format o)"
        & $py -m scripts.analysis.compute_symbolic_eval `
            --fit "fit_quick_${tag}.json" --out "eval_${tag}_self.json"
        Write-Output "[stageB] eval-full start $tag $(Get-Date -Format o)"
        & $py -m scripts.analysis.compute_symbolic_eval `
            --fit "fit_quick_${tag}.json" --out "eval_${tag}_full.json" --eval full
      }
      Write-Output "[stageB] done cfg=$cname norm=$norm ok=$? $(Get-Date -Format o)"
    }
  }
} finally {
  Stop-Transcript
}
