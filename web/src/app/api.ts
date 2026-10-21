import { HttpClient } from '@angular/common/http'
import { Injectable, inject } from '@angular/core'

export const API = 'http://localhost:8000'

export interface Sample {
  image: string
  label: string
}

export interface Explanation {
  method: string
  map: string
  stability: number
  badge: { level: 'trusted' | 'unstable' | 'not an explanation'; why: string }
}

export interface ExplainResult {
  notice: string
  prediction: string
  confidence: number
  explanations: Explanation[]
}

export interface Datasets {
  notice: string
  size: number
  datasets: Record<string, string[]>
}

@Injectable({ providedIn: 'root' })
export class Api {
  private http = inject(HttpClient)

  datasets() {
    return this.http.get<Datasets>(`${API}/datasets`)
  }

  samples(dataset: string) {
    return this.http.get<Sample[]>(`${API}/samples/${dataset}`)
  }

  async explain(dataset: string, dataUrl: string) {
    const blob = await (await fetch(dataUrl)).blob()
    const form = new FormData()
    form.append('image', blob, 'image.png')
    return this.http.post<ExplainResult>(`${API}/explain/${dataset}`, form)
  }

  drift(dataset: string) {
    return this.http.get<Record<string, { accuracy: number; drifted_columns: string[] }>>(`${API}/drift/${dataset}`)
  }
}
