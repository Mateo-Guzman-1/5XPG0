# Held-out confusable-word probe: synthesize words with the offline Windows
# voices (David, Zira) at 4 rates x 3 pitches into build/tts_probe/*.wav.
# These clips are never used for training; confusables.py scores them.
# Voices differ between Windows installs, so regenerated clips (and the probe
# numbers) can differ from the recorded results on another machine.
param([string]$Out = (Join-Path $PSScriptRoot 'build\tts_probe'))
New-Item -ItemType Directory -Force $Out | Out-Null
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$words = @('yes','yes!','yeets','yeet','yets','yetz','pizza','pizzas','yech','yetch','peach','each',
           'eats','jets','gets','bets','lets','yeah','sets','its','cheese','yesterday','less','guess')
$count = 0
foreach ($voice in @('Microsoft David Desktop','Microsoft Zira Desktop')) {
    $synth.SelectVoice($voice)
    foreach ($word in $words) { foreach ($rate in @('x-slow','slow','medium','fast')) { foreach ($pitch in @('low','medium','high')) {
        $name = ($word -replace '[^a-z]','') + '_' + $voice.Split(' ')[1] + "_$rate" + "_$pitch" + $(if ($word -match '!') {'_x'} else {''})
        $synth.SetOutputToWaveFile((Join-Path $Out "$name.wav"), $format)
        $synth.SpeakSsml("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><prosody rate='$rate' pitch='$pitch'>$word</prosody></speak>")
        $count++
    }}}
}
$synth.SetOutputToNull()
"$count clips -> $Out"
