import data from '@/data/models.json'
import photos from '@/data/photos.json'

export type Model = (typeof data.models)[number]
export type Photo = (typeof photos)[keyof typeof photos]

export const models: Model[] = data.models
export const groups = data.groups as Record<string, string>
export const packageVersion = data.version
export const photoFor = (key: string): Photo => (photos as Record<string, Photo>)[key]
export const photoCredits = photos as Record<string, Photo>

export const evidenceLabel = (e: string) =>
  e === 'A' ? 'Verified against a reference implementation' : 'Checked against the published paper'
