import { Component, computed, inject, signal } from '@angular/core'
import { DecimalPipe, KeyValuePipe } from '@angular/common'
import { Api, Datasets, ExplainResult, Sample } from './api'

const LABELS: Record<string, string> = {
  gradient: 'Gradient',
  smoothgrad: 'SmoothGrad',
  integrated_gradients: 'Integrated Gradients',
  guided_backprop: 'Guided Backprop',
  gradcam: 'Grad-CAM',
  guided_gradcam: 'Guided Grad-CAM',
}

@Component({
  selector: 'app-root',
  imports: [DecimalPipe, KeyValuePipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private api = inject(Api)
  labels = LABELS
  info = signal<Datasets | null>(null)
  dataset = signal('pneumoniamnist')
  samples = signal<Sample[]>([])
  current = signal<Sample | null>(null)
  result = signal<ExplainResult | null>(null)
  method = signal('gradcam')
  opacity = signal(0.55)
  busy = signal(false)
  error = signal('')
  drift = signal<Record<string, { accuracy: number; drifted_columns: string[] }>>({})

  shown = computed(() => this.result()?.explanations.find((e) => e.method === this.method()) ?? null)

  constructor() {
    this.api.datasets().subscribe({ next: (d) => this.info.set(d), error: () => this.error.set('API not reachable') })
    this.load('pneumoniamnist')
  }

  load(dataset: string) {
    this.dataset.set(dataset)
    this.result.set(null)
    this.current.set(null)
    this.api.samples(dataset).subscribe((s) => {
      this.samples.set(s)
      if (s.length) this.pick(s[0])
    })
    this.api.drift(dataset).subscribe((d) => this.drift.set(d))
  }

  async pick(sample: Sample) {
    this.current.set(sample)
    this.result.set(null)
    this.busy.set(true)
    ;(await this.api.explain(this.dataset(), sample.image)).subscribe({
      next: (r) => {
        this.result.set(r)
        this.busy.set(false)
      },
      error: () => {
        this.error.set('Could not explain this image')
        this.busy.set(false)
      },
    })
  }

  upload(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => this.pick({ image: reader.result as string, label: 'unknown (your image)' })
    reader.readAsDataURL(file)
  }
}
