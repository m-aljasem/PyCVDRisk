import data from '@/data/models.json'
import photos from '@/data/photos.json'

export type Model = (typeof data.models)[number]
export type Photo = (typeof photos)[keyof typeof photos]

export const models: Model[] = data.models
export const groups = data.groups as Record<string, string>
export const packageVersion = data.version
export const photoFor = (key: string): Photo => (photos as Record<string, Photo>)[key]
export const photoCredits = photos as Record<string, Photo>

// Evidence levels (manuscript): A developers' code/calculator/charts, B independent package, C the paper's own examples and tables
export const referenceVerified = (e: string) => /[AB]/.test(e)
export const evidenceLabel = (e: string) =>
  `${referenceVerified(e) ? 'Verified against a reference implementation' : 'Checked against the published paper'} (evidence ${e})`
